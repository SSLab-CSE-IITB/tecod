"""Main TeCoD service orchestrating all components."""

import time
from typing import NamedTuple

import pandas as pd

from ..exceptions.base import GenerationError, ServiceInitializationError
from ..models.data import GenerationOutput, GenerationRequest, TemplateSelectionResult
from ..nli import NLI
from ..pdec.partitioned_decoding import partitioned_decoding
from ..prompts import generate_prompt
from ..utils.generation import calculate_log_prob, get_data, get_gen_sequences
from ..utils.timing import log_with_time_elapsed
from .base import ModelServiceProtocol, Service
from .embedding import EmbeddingService
from .template import TemplateService
from .vector_store import VectorStoreService


class _GenerationResult(NamedTuple):
    """Result from a generation method (ICL or API)."""

    text: str
    log_prob: float | None = None
    log_logits_prob: float | None = None


def pick_icl_example_indices(retrieved_examples: pd.DataFrame, icl_cnt: int) -> list[int]:
    """Pick the best retrieved example index per template for ICL."""
    if icl_cnt <= 0:
        return []

    example_ids = (
        retrieved_examples["id"]
        if "id" in retrieved_examples.columns
        else retrieved_examples.index.to_series(index=retrieved_examples.index)
    )
    ranked_examples = retrieved_examples.assign(_example_id=example_ids).sort_values(
        "cosine_score", ascending=False, kind="mergesort"
    )
    best_per_template = ranked_examples.drop_duplicates("t_id", keep="first")
    return best_per_template["_example_id"].head(icl_cnt).astype(int).to_list()


class TeCoDService(Service):
    """
    Main TeCoD service that orchestrates all components for text-to-SQL generation.

    This service implements the Template Constrained Decoding (TeCoD) approach
    for converting natural language queries into SQL statements. It combines vector similarity
    search, natural language inference (NLI), and constraint-based decoding to generate
    accurate SQL queries.

    The service supports multiple generation methods:
    - **GCD** (Guided Constrained Decoding): Uses partitioned decoding for efficient generation — only literal slots are generated, template structure is reused
    - **Base-GCD**: Converts template grammar to regex and passes it to outlines for whole-sequence constrained generation
    - **SGC** (Soft Grammar Constraining): Includes matched template in the prompt as soft guidance for unconstrained generation
    - **ICL** (In-Context Learning): Unconstrained generation with in-context examples, no template
    - **ZS** (Zero-Shot): Unconstrained generation without examples or templates

    Attributes:
        embedding_service (EmbeddingService): Service for query vectorization
        vector_store_service (VectorStoreService): Service for similarity search
        model_service (ModelService): Service for language model inference
        template_service (TemplateService): Service for SQL template management
        device (str): Device used for computation ('cuda' or 'cpu')

    Example:
        >>> tecod_service = TeCoDService(config, embedding_service, vector_store_service,
        ...                            model_service, template_service)
        >>> tecod_service.initialize()
        >>> request = GenerationRequest(
        ...     query="How many accounts from north Bohemia are eligible to receive loans?"
        ... )
        >>> result = tecod_service.generate(request)
        >>> print(result.pred_sql)
        SELECT COUNT(T1.account_id) ...
    """

    def __init__(
        self,
        config,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        model_service: ModelServiceProtocol,
        template_service: TemplateService,
        device: str | None = None,
        logger=None,
    ):
        super().__init__(config, logger)
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.model_service = model_service
        self.template_service = template_service

        self.device = device

        # Will be initialized in initialize()
        self._nli = None
        self._examples_df: pd.DataFrame | None = None
        self._schema_prompt: str | None = None

    @property
    def _is_api_model(self) -> bool:
        """Check if the model service is an API-based model."""
        return self.config.tecod.is_api_model

    def initialize(self) -> None:
        """Initialize the TeCoD service and all dependencies."""
        try:
            with log_with_time_elapsed("TeCoD service initialization", self.logger):
                self.logger.info("Initializing TeCoD service...")

                # Initialize all sub-services
                services_to_init = [
                    ("embedding", self.embedding_service),
                    ("vector_store", self.vector_store_service),
                    ("model", self.model_service),
                    ("template", self.template_service),
                ]

                for service_name, service in services_to_init:
                    if not service.is_initialized:
                        self.logger.info(f"Initializing {service_name} service...")
                        service.initialize()

                # Load examples data
                with log_with_time_elapsed("Loading examples data", self.logger):
                    examples_path = self.config.examples_path
                    if not examples_path.exists():
                        raise ServiceInitializationError(
                            "TeCoDService", f"Examples file not found: {examples_path}"
                        )

                    self._examples_df = pd.read_json(examples_path, lines=True)
                    self._validate_examples_data(self._examples_df)
                    self.logger.info(f"Loaded {len(self._examples_df)} examples")

                # Load schema prompt
                with log_with_time_elapsed("Loading schema prompt", self.logger):
                    schema_path = self.config.schema_prompt_path
                    if not schema_path.exists():
                        raise ServiceInitializationError(
                            "TeCoDService",
                            f"Schema prompt file not found: {schema_path}",
                        )

                    with open(schema_path) as f:
                        self._schema_prompt = f.read()

                # Initialize NLI model
                with log_with_time_elapsed("Initializing NLI model", self.logger):
                    self.logger.info("Initializing NLI model...")
                    self._nli = NLI(model_id=self.config.nli.model, device=self.device)

                self._mark_initialized()
                self.logger.info("TeCoD service initialized successfully")

        except Exception as e:
            self.logger.exception(f"Failed to initialize TeCoDService: {str(e)}")
            raise ServiceInitializationError("TeCoDService", str(e)) from e

    def cleanup(self) -> None:
        """Cleanup all resources."""
        services_to_cleanup = [
            self.embedding_service,
            self.vector_store_service,
            self.model_service,
            self.template_service,
        ]

        for service in services_to_cleanup:
            try:
                service.cleanup()
            except Exception as e:
                self.logger.exception(f"Error during cleanup: {e}")

        self._nli = None
        self._examples_df = None
        self._schema_prompt = None
        self._initialized = False

    def _get_oracle_template(self, question, sql_query):
        # check if template exists
        if self._is_api_model:
            raise GenerationError(
                self.config.tecod.model_id,
                "Oracle template compilation requires direct model access "
                "and is not supported with API models.",
            )

        from sqlglot import parse_one
        from sqlglot.optimizer.optimize_joins import optimize_joins

        from ..pdec.compile_template import generate_token_ids_and_save_to_store

        template = optimize_joins(parse_one(sql_query, dialect=self.config.tecod.dialect)).sql(
            dialect=self.config.tecod.dialect
        )

        # check if template exists in templates
        templates = self.template_service.get_all_templates()
        templates = templates[templates["template"] == template]

        if not templates.empty:
            template_id = templates.index[0]
            compiled_template = self.template_service.load_compiled_template(template_id)
            return template, template_id, compiled_template
        else:
            template_id = -1

        # Get ICL examples (simplified version)
        search_results = self.vector_store_service.search(
            question, top_k=self.config.tecod.vectorsearch_top_k
        )
        retrieved_examples = self.examples.iloc[search_results[0].index].copy()
        retrieved_examples["cosine_score"] = search_results[0]["distance"].values

        icl_example_indices = pick_icl_example_indices(
            retrieved_examples, self.config.tecod.icl_cnt
        )

        icl_examples = []
        if self.config.tecod.icl_cnt > 0:
            for _, ex_row in self.examples.iloc[icl_example_indices].iterrows():
                icl_examples.append((ex_row["text"], ex_row[self.config.tecod.sql_key]))
            icl_examples = icl_examples[::-1]

        # Generate prompt
        prompt = generate_prompt(
            model_id=self.config.tecod.model_id,
            prompt_class=self.config.tecod.prompt_class or None,
            schema_sequence=self.schema_prompt,
            content_sequence="",
            question_text=question,
            icl_examples=icl_examples,
            database_engine=self.config.tecod.dialect,
        )

        compiled_template = generate_token_ids_and_save_to_store(
            model=self.model_service.model,
            template_id=-1,
            tokenizer=self.model_service.tokenizer,
            prompt=prompt,
            sql_query=sql_query,
            db_path=str(self.config.db_file_path),
            ebnf_type=self.config.tecod.grammar_type,
        )

        return template, template_id, compiled_template[template_id]

    def generate_with_method(self, request: GenerationRequest) -> GenerationOutput:
        """Generate SQL from natural language query with explicit method selection.

        This method supports explicit generation method selection for analysis
        and comparison. For routine use, prefer the `generate()` method
        which automatically selects the best method based on NLI classification.

        Supported methods:
            - 'gcd': Partitioned decoding — generates only literal slots for efficient template-based generation
            - 'base-gcd': Template grammar converted to regex and passed to outlines for constrained generation
            - 'sgc': Template included in prompt as soft guidance for unconstrained generation
            - 'icl': Unconstrained generation with in-context examples, no template
            - 'zs': Unconstrained generation without examples or templates
            - 'auto': Automatic method selection based on NLI

        Args:
            request: Generation request with explicit method specified

        Returns:
            Complete generation output with detailed timing data
        """
        self.ensure_initialized()

        # Caller's request must not be mutated. Copy before any field writes.
        request = request.model_copy()

        # Validate method compatibility with API models
        if self._is_api_model and request.method in ("gcd", "base-gcd"):
            raise ValueError(
                f"Method '{request.method}' requires direct model logit access "
                f"and is not supported with API models (provider={self.config.tecod.provider}). "
                f"Use 'sgc', 'icl', or 'zs' instead."
            )

        if request.method == "auto":
            request.schema_sequence = None
            request.content_sequence = None
            request.use_oracle = False
            request.gold_sql = None

        # Start overall timing
        overall_start_time = time.perf_counter()
        timing_data = {}
        pipeline_metrics = {}

        query_preview = f"{request.query[:50]}{'...' if len(request.query) > 50 else ''}"
        self.logger.info(f"Starting TeCoD generation for query: '{query_preview}'")

        # Template selection with detailed timing
        if request.method != "zs":
            template_start = time.perf_counter()
            template_result = self._template_selection_with_timing(
                request.query, request.top_k, timing_data, pipeline_metrics
            )
            template_selection_time = time.perf_counter() - template_start
            timing_data["template_selection_total"] = template_selection_time
        else:
            template_result = TemplateSelectionResult(
                template_id=-1,
                entailment_score=0.0,
                cosine_score=0.0,
                nli_label="none",
                icl_examples=[],
                icl_example_indices=[],
            )
            template_selection_time = 0.0
            timing_data["template_selection_total"] = 0.0

        template = None
        template_id = template_result.template_id
        compiled_template = None
        template_methods = {"auto", "sgc", "gcd", "base-gcd"}
        has_usable_template = (
            request.method in template_methods
            and template_result.template_id >= 0
            and template_result.nli_label == "entailment"
        )
        if request.use_oracle:
            # first check if gold template exists or not
            template, template_id, compiled_template = self._get_oracle_template(
                request.query, request.gold_sql
            )
            has_usable_template = True
        elif has_usable_template:
            template = self.template_service.get_template_info(template_id)["template"]

        # Generate prompt
        prompt_start = time.perf_counter()
        prompt = generate_prompt(
            model_id=self.config.tecod.model_id,
            prompt_class=self.config.tecod.prompt_class or None,
            schema_sequence=request.schema_sequence or self._schema_prompt,
            content_sequence=request.content_sequence or "",
            question_text=request.query,
            icl_examples=template_result.icl_examples if request.method != "zs" else [],
            template=template if request.method == "sgc" and has_usable_template else None,
            database_engine=self.config.tecod.dialect,
        )
        timing_data["prompt_generation"] = time.perf_counter() - prompt_start

        # Choose generation method based on NLI result
        generation_start = time.perf_counter()

        if has_usable_template:
            # Template selected
            self.logger.info("Using template")

            if request.method == "sgc" or (self._is_api_model and request.method == "auto"):
                sgc_start = time.perf_counter()
                if self._is_api_model:
                    generation_result = self._generate_with_api(prompt, request)
                else:
                    generation_result = self._generate_with_icl(prompt, request)
                timing_data["sgc_generation"] = time.perf_counter() - sgc_start
                pred_sql = generation_result.text
                log_prob = generation_result.log_prob
                log_logits_prob = generation_result.log_logits_prob
                method = "sgc"
            elif request.method == "base-gcd":
                from ..pdec.tecod_utils import convert_template_to_ebnf, ebnf_to_regex

                sql_ebnf, new_rules_for_ebnf = convert_template_to_ebnf(
                    template=template,
                    remove_aliases=False,
                    db_path=self.config.db_file_path,
                    type=self.config.tecod.grammar_type,
                )
                # tab_and_col_to_rule_grammar, base_grammar
                sql_regex_grammar = ebnf_to_regex(
                    ebnf_str=sql_ebnf, new_rules_for_ebnf=new_rules_for_ebnf
                )
                request.regex_grammar = sql_regex_grammar
                gcd_start = time.perf_counter()
                generation_result = self._generate_with_icl(prompt, request)
                timing_data["base-gcd_generation"] = time.perf_counter() - gcd_start
                pred_sql = generation_result.text
                log_prob = generation_result.log_prob
                log_logits_prob = generation_result.log_logits_prob
                method = "base-gcd"
            else:
                gcd_start = time.perf_counter()
                pred_sql = self._generate_with_template(prompt, template_id, compiled_template)
                timing_data["gcd_generation"] = time.perf_counter() - gcd_start
                method = "gcd"
                log_prob = None
                log_logits_prob = None
        else:
            # Unconstrained generation [icl, zs]
            self.logger.info("Using in-context learning (ICL)")
            icl_start = time.perf_counter()
            if self._is_api_model:
                generation_result = self._generate_with_api(prompt, request)
            else:
                generation_result = self._generate_with_icl(prompt, request)
            fallback_method = "zs" if request.method == "zs" else "icl"
            timing_data[f"{fallback_method}_generation"] = time.perf_counter() - icl_start
            pred_sql = generation_result.text
            log_prob = generation_result.log_prob
            log_logits_prob = generation_result.log_logits_prob
            method = fallback_method

        generation_time = time.perf_counter() - generation_start
        timing_data["generation_total"] = generation_time

        # Post-process SQL
        postprocess_start = time.perf_counter()
        post_processing_failed = False
        try:
            pred_sql = self._post_process_sql(pred_sql)
        except Exception as e:
            self.logger.exception(f"Post-processing error: {e}")
            self.logger.error(f"Raw generation: {pred_sql}")
            post_processing_failed = True
        timing_data["post_processing"] = time.perf_counter() - postprocess_start

        # Calculate total time
        total_time = time.perf_counter() - overall_start_time
        timing_data["total_time"] = total_time

        # Log timing summary
        self.logger.info(f"TeCoD generation completed in {total_time * 1000:.1f}ms")
        timing_data_ms = {
            k: f"{v * 1000:.1f}ms" if isinstance(v, (int, float)) else v
            for k, v in timing_data.items()
        }
        self.logger.debug(f"Timing breakdown: {timing_data_ms}")

        return GenerationOutput(
            query=request.query,
            pred_sql=pred_sql,
            method=method,
            template_id=template_result.template_id,
            nli_score=template_result.entailment_score,
            cosine_score=template_result.cosine_score,
            nli_label=template_result.nli_label,
            icl_examples=template_result.icl_examples,
            icl_example_indices=template_result.icl_example_indices,
            log_prob=log_prob,
            log_logits_prob=log_logits_prob,
            generation_time=generation_time,
            prompt=prompt,
            timing_data=timing_data,
            total_time=total_time,
            vector_search_time=timing_data.get("vector_search"),
            nli_processing_time=timing_data.get("nli_processing"),
            template_selection_time=template_selection_time,
            embedding_time=timing_data.get("query_embedding"),
            retrieved_examples_count=pipeline_metrics.get("retrieved_examples_count"),
            nli_examples_count=pipeline_metrics.get("nli_examples_count"),
            # Retained for callers that need template-selection diagnostics.
            template_selection_result=template_result,
            post_processing_failed=post_processing_failed,
        )

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        """Generate SQL from natural language query.

        Delegates to generate_with_method() with method='auto', which automatically
        selects the best method based on NLI classification.

        Args:
            request: Generation request

        Returns:
            Complete generation output with detailed timing data
        """
        # model_copy(update=...) produces a new instance with method flipped
        # to "auto"; the caller's request object is never touched. Without
        # this, `request.method = "auto"` would mutate the caller's object
        # before generate_with_method's defensive copy can take effect.
        return self.generate_with_method(request.model_copy(update={"method": "auto"}))

    def _template_selection_with_timing(
        self, query: str, top_k: int, timing_data: dict, pipeline_metrics: dict
    ) -> TemplateSelectionResult:
        """Select the best template for the given query with detailed timing.

        Args:
            query: Natural language query
            top_k: Number of top results to consider
            timing_data: Dictionary to store timing durations
            pipeline_metrics: Dictionary to store pipeline counts

        Returns:
            Template selection result
        """
        # Vector search with timing
        search_start = time.perf_counter()
        search_results = self.vector_store_service.search(query, top_k, timing_data=timing_data)
        timing_data["vector_search"] = time.perf_counter() - search_start

        retrieved_examples = self._examples_df.iloc[search_results[0].index].copy()
        retrieved_examples["cosine_score"] = search_results[0]["distance"].values
        retrieved_examples["id"] = retrieved_examples.index
        pipeline_metrics["retrieved_examples_count"] = len(retrieved_examples)

        # NLI filtering with timing
        nli_start = time.perf_counter()
        nli_results = self._perform_nli_with_timing(
            retrieved_examples, query, timing_data, pipeline_metrics
        )
        timing_data["nli_processing"] = time.perf_counter() - nli_start

        t_groups = nli_results.groupby("t_id")

        # Get ICL examples
        icl_start = time.perf_counter()
        icl_examples, icl_example_indices = self._get_icl_examples(retrieved_examples)
        timing_data["icl_preparation"] = time.perf_counter() - icl_start

        # Select best template
        selection_start = time.perf_counter()
        aggregation_method = self.config.nli.method
        templates_considered = t_groups.agg(
            {"entailment": aggregation_method, "cosine_score": aggregation_method}
        ).sort_values(["entailment", "cosine_score"], ascending=False)

        if templates_considered.empty:
            self.logger.warning(
                "No templates survived NLI filtering, falling back to unconstrained generation"
            )
            timing_data["template_scoring"] = time.perf_counter() - selection_start
            return TemplateSelectionResult(
                template_id=-1,
                entailment_score=0.0,
                cosine_score=0.0,
                nli_label="none",
                icl_examples=icl_examples,
                icl_example_indices=icl_example_indices,
            )

        t_row = templates_considered.iloc[0]

        t_score = t_row.entailment
        t_id = t_row.name
        t_cosine_score = t_row.cosine_score
        t_nli_label = (
            nli_results[nli_results["t_id"] == t_id]
            .value_counts("nli_label", ascending=False)
            .index[0]
        )
        timing_data["template_scoring"] = time.perf_counter() - selection_start

        return TemplateSelectionResult(
            template_id=t_id,
            entailment_score=t_score,
            cosine_score=t_cosine_score,
            nli_label=t_nli_label,
            icl_examples=icl_examples,
            icl_example_indices=icl_example_indices,
            retrieved_examples=retrieved_examples,
            nli_results=nli_results,
            templates_considered=templates_considered,
        )

    def _perform_nli_with_timing(
        self,
        retrieved_examples: pd.DataFrame,
        query: str,
        timing_data: dict,
        pipeline_metrics: dict,
    ) -> pd.DataFrame:
        """Perform NLI classification on retrieved examples with timing.

        Args:
            retrieved_examples: Retrieved examples DataFrame
            query: Query text
            timing_data: Dictionary to store timing durations
            pipeline_metrics: Dictionary to store pipeline counts

        Returns:
            NLI results DataFrame
        """
        retrieved_examples = retrieved_examples.copy()
        retrieved_examples = retrieved_examples[: self.config.tecod.nli_top_k]
        pipeline_metrics["nli_examples_count"] = len(retrieved_examples)

        # Extract NLQs for NLI
        extract_start = time.perf_counter()
        nlqs = retrieved_examples[self.config.emb.masked_nlq_key].tolist()
        timing_data["nli_data_extraction"] = time.perf_counter() - extract_start

        # Perform NLI inference
        inference_start = time.perf_counter()
        nli_output = self._nli(nlqs, query)
        timing_data["nli_inference"] = time.perf_counter() - inference_start

        # Process NLI results
        processing_start = time.perf_counter()
        nli_df = pd.DataFrame(nli_output)
        nli_df["nli_label"] = nli_df.apply(lambda x: x.index[x.argmax()], axis=1)
        nli_df["id"] = retrieved_examples.index
        result = nli_df.merge(retrieved_examples, how="left", left_on="id", right_index=True)
        timing_data["nli_result_processing"] = time.perf_counter() - processing_start

        return result

    def _get_icl_examples(
        self, retrieved_examples: pd.DataFrame
    ) -> tuple[list[tuple[str, str]], list[int]]:
        """Get in-context learning examples.

        Args:
            retrieved_examples: Retrieved examples DataFrame

        Returns:
            Tuple of (ICL examples, ICL example indices)
        """
        icl_example_indices = pick_icl_example_indices(
            retrieved_examples, self.config.tecod.icl_cnt
        )

        icl_examples = []
        if self.config.tecod.icl_cnt > 0:
            for _, row in self._examples_df.iloc[icl_example_indices].iterrows():
                icl_nlq = row["text"]
                icl_sql = row[self.config.tecod.sql_key]
                icl_examples.append((icl_nlq, icl_sql))
            # Reverse to keep most similar example close to the question
            icl_examples = icl_examples[::-1]

        return icl_examples, icl_example_indices

    def _generate_with_template(
        self, prompt: str, template_id: int, compiled_template: dict | None = None
    ) -> str:
        """Generate SQL using template-guided constrained decoding.

        Uses the partitioned decoding algorithm: the compiled template is split
        into fixed structural parts and variable literal slots, each constrained
        by a dedicated regex logit processor.

        Args:
            prompt: Generation prompt
            template_id: Template ID to use
            compiled_template: Pre-compiled template dictionary (if available)

        Returns:
            Generated SQL string
        """
        if compiled_template is None:
            compiled_template = self.template_service.load_compiled_template(template_id)

        return partitioned_decoding(
            prompt=prompt,
            model=self.model_service.model,
            tokenizer=self.model_service.tokenizer,
            device=self.device,
            template=compiled_template,
            template_id=template_id,
        )

    def _generate_with_icl(self, prompt: str, request: GenerationRequest) -> _GenerationResult:
        """Generate SQL using in-context learning.

        Args:
            prompt: Generation prompt
            request: Generation request

        Returns:
            Generation result with text and log probabilities
        """
        inputs = get_data(prompts=[prompt], tokenizer=self.model_service.tokenizer)

        output = self.model_service.generate(
            inputs=inputs,
            max_new_tokens=request.max_new_tokens,
            num_beams=request.num_beams,
            do_sample=request.do_sample,
            regex_grammar=request.regex_grammar,
        )

        # Process generation results
        gen_sequences = get_gen_sequences(
            sequences=output.sequences,
            tokenizer=self.model_service.tokenizer,
            inputs=inputs,
        )

        log_prob = calculate_log_prob(
            logits=output.scores,
            gen_sequence=gen_sequences,
            device=self.device,
        )

        log_logits_prob = calculate_log_prob(
            logits=output.logits,
            gen_sequence=gen_sequences,
            device=self.device,
        )

        generations = self.model_service.batch_decode(
            gen_sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return _GenerationResult(
            text=generations[0],
            log_prob=log_prob[0] if log_prob else None,
            log_logits_prob=log_logits_prob[0] if log_logits_prob else None,
        )

    def _generate_with_api(self, prompt: str, request: GenerationRequest) -> _GenerationResult:
        """Generate SQL using an API-based model.

        Args:
            prompt: Generation prompt
            request: Generation request

        Returns:
            Generation result with text only (no log probabilities for API models)

        Raises:
            GenerationError: If the API returns None or empty text
        """
        text = self.model_service.generate_sql(
            prompt=prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=getattr(self.config.tecod, "temperature", 0.0),
        )
        if not text:
            raise GenerationError(
                self.config.tecod.model_id,
                "API returned empty text",
            )
        return _GenerationResult(text=text)

    @staticmethod
    def _post_process_sql(sql: str | None) -> str | None:
        """Post-process generated SQL.

        Extracts the SQL out of any fenced code blocks the model emitted
        and returns only the first statement, terminated with a
        semicolon. When the model output parses as a single statement
        (the common case), the original text is returned verbatim so
        that exact-match comparisons are not perturbed by sqlglot's whitespace and keyword-case
        normalisation. Only when the model emitted multiple statements
        do we fall through to sqlglot's re-emission of the first one.
        """
        if not sql:
            return sql
        while "```sql" in sql:
            sql = sql.split("```sql")[-1].split("```")[0].strip("\n").strip()

        import sqlglot
        from sqlglot.errors import ParseError

        try:
            statements = sqlglot.parse(sql, dialect="sqlite")
        except ParseError:
            statements = None

        if statements and len(statements) == 1:
            # Single statement: preserve the model's exact text (minus
            # trailing ';' / whitespace), append one terminator.
            return sql.strip().rstrip(";").rstrip() + ";"
        if statements and statements[0] is not None:
            # Multiple statements parsed — keep only the first. Accept
            # sqlglot's normalisation here; the alternative is a
            # character-offset slice which sqlglot does not expose.
            return statements[0].sql(dialect="sqlite") + ";"
        # Unparseable: preserve prior behaviour (naive split).
        return sql.split(";")[0].strip() + ";"

    @property
    def examples(self) -> pd.DataFrame:
        """Get examples DataFrame."""
        self.ensure_initialized()
        return self._examples_df

    @property
    def schema_prompt(self) -> str:
        """Get schema prompt."""
        self.ensure_initialized()
        return self._schema_prompt

    def _validate_examples_data(self, examples: pd.DataFrame) -> None:
        """Validate prepared examples before generation starts."""
        required_columns = [
            "text",
            self.config.tecod.sql_key,
            self.config.emb.masked_nlq_key,
            "t_id",
        ]
        missing = [column for column in required_columns if column not in examples.columns]
        if missing:
            raise ServiceInitializationError(
                "TeCoDService",
                "Examples file is missing required column(s): "
                f"{', '.join(missing)}. Required columns: "
                f"{', '.join(required_columns)}.",
            )
