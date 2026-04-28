import torch
import torch.nn.functional as F


def get_data(*, prompts: list, tokenizer, padding_side="left"):
    if isinstance(prompts, str):
        prompts = [prompts]
    inputs = tokenizer(
        prompts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=False,
        return_length=True,
        padding_side=padding_side,
    )
    if padding_side == "left":
        length = inputs["length"] - inputs["attention_mask"].argmax(axis=1)
    else:
        length = inputs["attention_mask"].argmin(axis=1)
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "length": length,
    }


def get_gen_sequences(*, sequences, tokenizer, inputs):
    output_seq_start_end = []
    pad_token = tokenizer.pad_token_id
    for i in range(sequences.shape[0]):
        nonzeros = (sequences[i] != pad_token).nonzero()
        if nonzeros.numel() == 0:
            output_seq_start_end.append(None)
            continue
        output_seq_start_end.append((nonzeros[0][0], nonzeros[-1][0]))

    gen_sequence = []
    for i, (span, length) in enumerate(zip(output_seq_start_end, inputs["length"])):
        if span is None:
            gen_sequence.append(sequences[i, :0])  # empty tensor
            continue
        start, end = span
        gen_sequence.append(sequences[i, start + length : end + 1])

    return gen_sequence


def calculate_log_prob(*, logits, gen_sequence, device):
    logits = torch.stack(logits, 1).to(device)
    log_logits = F.log_softmax(logits, dim=-1)

    log_prob = []
    for i, sequence in enumerate(gen_sequence):
        log_prob.append(
            torch.sum(torch.gather(log_logits[i], -1, sequence.unsqueeze(-1)))
            .detach()
            .cpu()
            .numpy()
            .item()
        )

    return log_prob


def post_process(*, outputs):
    results = []
    for (
        sql,
        generation,
        log_prob,
        log_logits_prob,
    ) in outputs:
        if not sql or not sql.strip():
            results.append(
                {
                    "generation": generation,
                    "sql": "",
                    "log_prob": log_prob,
                    "log_logits_prob": log_logits_prob,
                }
            )
            continue

        # sql = sql.replace("\n", " ")
        while "``" in sql:
            sql = sql.replace("``", "`")

        sql = sql.split(";")[0].strip() + ";"

        results.append(
            {
                "generation": generation,
                "sql": sql,
                "log_prob": log_prob,
                "log_logits_prob": log_logits_prob,
            }
        )
    return results
