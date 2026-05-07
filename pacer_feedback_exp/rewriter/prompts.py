from __future__ import annotations


def basic_prompt(sentences) -> str:
    if isinstance(sentences, str):
        return sentences
    return " ".join(str(x) for x in sentences)



def make_instruct_stitch_prompt(bin_payloads, score_col: str = "utility") -> str:
    high = bin_payloads["high"]
    mid = bin_payloads["mid"]
    low = bin_payloads["low"]

    low_summary = low["summary"].strip() if low["summary"] else "No important low-relevance caveats."

    return f"""
Task: Write a final personalized review summary using the cluster data below.

Rules:
1. Write exactly 4 sentences in total.
2. Sentence 1 must describe the HIGH cluster.
3. Sentence 2 must describe the MID cluster.
4. Sentence 3 must describe the LOW cluster.
5. Sentence 4 must give an overall recommendation.
6. Do not copy the input text verbatim.

Input:
HIGH | pct={high['stats']['pct']:.1f}% | count={high['stats']['count']} | mean_{score_col}={high['stats']['mean_score']:.3f} | summary={high['summary']}
MID  | pct={mid['stats']['pct']:.1f}% | count={mid['stats']['count']} | mean_{score_col}={mid['stats']['mean_score']:.3f} | summary={mid['summary']}
LOW  | pct={low['stats']['pct']:.1f}% | count={low['stats']['count']} | mean_{score_col}={low['stats']['mean_score']:.3f} | summary={low_summary}

Output:
""".strip()



def make_final_third_person_prompt(final_summary_text: str) -> str:
    return f"""
Rewrite the following summary so it is suitable to display to users.

Rules:
- Use third-person generalization.
- Remove first-person phrases like "I", "me", "my", "we".
- Do not sound like direct review quotes.
- Keep all statistics and percentages unchanged.
- Preserve the overall meaning.
- Return one polished paragraph.

Summary:
{final_summary_text}

Rewritten summary:
""".strip()
