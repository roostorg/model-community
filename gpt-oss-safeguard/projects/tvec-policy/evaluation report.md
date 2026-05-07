# Policy Evaluation Results: gpt-oss-safeguard

**Last updated: 17 April 2026 · n=170**

---

## 1. Overall Metrics

| Metric | Value | Target |
|---|---|---|
| Precision | **100.0%** | ≥90% | 
| Recall | **90.5%** | ≥90% | 
| F1 Score | **95.0%** | ≥90% | 
| False Positive Rate | **0.0%** | ≤5% | 
| False Negative Rate | **9.5%** | ≤10% | 
| Specificity | **100.0%** | ≥90% | 
| Cohen's Kappa | **0.831** | ≥0.75 | 
| MCC | **0.843** | ≥0.80 | 
| TVEC2 Recall *(Credible threat to life)* | **100.0%** | ≥95% |

---

## 2. Confusion Matrix

|  | **Model: Invalid (remove)** | **Model: Valid (permit)** |
|---|---|---|
| **Actual: Invalid** | TP = 114 | FN = 12 |
| **Actual: Valid** | FP = 0 | TN = 44 |

`TP=114 · TN=44 · FP=0 · FN=12 · Total=170`

---

## 3. Per-Category Performance

### Recall (and correct permit rate for valid categories)

| Category | Metric | Target | Value | TP or TN / Total | Status |
|---|---|---|---|---|---|
| TVEC1 | Recall | ≥85% | **100%** | 15 / 15 | ✅ Pass |
| TVEC2 | Recall | ≥95% | **100%** | 17 / 17 | ✅ Pass |
| TVEC3 | Recall | ≥85% | **92%** | 12 / 13 | ✅ Pass |
| TVEC4 | Recall | ≥85% | **85%** | 11 / 13 | ✅ Pass |
| TVEC5 | Recall | ≥95% | **100%** | 14 / 14 | ✅ Pass |
| TVEC6 | Recall | ≥85% | **71%** | 12 / 17 | ❌ Fail |
| TVEC7 | Recall | ≥85% | **93%** | 13 / 14 | ✅ Pass |
| TVEC8 | Recall | ≥85% | **87%** | 20 / 23 | ✅ Pass |
| TVEC0 | Correct permit rate | ≥90% | **100%** | 30/30 | ✅ Pass |
| TVEC.01 | Correct permit rate | ≥90% | **100%** | 14 / 14 | ✅ Pass |

Note that recall targets vary across content categories. The content categories with the greatest potential harm (TVEC2 - Credible threat to life and TVEC5 - Instructional material to commit a terrorist attack) are held to a higher performance standard (recall ≥95%). Permissible content categories (TVEC0 - Non-TVEC and TVEC.01 - EDSA & Newsworthiness Exceptions) are also held to a higher performance standard (recall ≥90%), given the impact of over-removal on freedom of expression. All other categories are evaluated based on a target of ≥85% recall. 

---

## 4. Failure Cluster Analysis

| Failure Cluster | Categories Impacted | Count | Interpretation |
|---|---|---|---|
| Entity recognition gap | TVEC4, TVEC6 | 5 | Model fails when content references lesser known terrorist and violent extremist organisations. Addressing this gap requires fine-tuning or a supplementary reference list (like that provided in Annex 2. |
| Threshold not met | TVEC6, TVEC7, TVEC8 | 4 | Model recognises potential harm but concludes it does not meet the threshold for terrorism or violent extremism; whereas experts on our team disagreed based on our interpretation and application of the policy. This misalignment affects content in the test dataset related to violent accelerationism and violence targeting women and immigrants. Future policy refinement could seek to close this gap. |
| Framing confusion | TVEC3, TVEC4 | 2 | Operational content framed as advice or material support framed as community events or political activities were not flagged. In these examples, it appears the model was influenced by surface framing and overlooked details that provide important context. |
| Public figures | TVEC8 | 1 | In one case, the model did not recognize a threat against a public political figure as incitement to terrorism or violent extremism. This could be improved through future policy refinement. |

---

## 5. Key Takeaways

1. **Strong Performance on Permissible Speech:** In this limited test, the model produced no false positives, correctly identifying all permitted speech as non-violative. This is an encouraging result, as avoiding false positives was a core design priority — over-restriction risks chilling legitimate discourse and free expression.

2. **Improving Entity Recognition:** Model performance could be improved if the policy is used alongside a reference list of terrorist and violent extremist actors (groups, individual perpetrators, and communities). Annex 2 provides a starting point for developing these lists, however we strongly recommend that users implementing this policy work directly with subject matter experts in preventing and countering terrorism and violent extremism to develop more fullsome internal resources based on latest available evidence.

4. **EDSA Precision:** Finally, we note that the models struggels to apply TVEC.01 (the EDSA exceptions) accurately, with a precision rate of just 72.2%. Importantly, none of the false negatives in the dataset were permitted under the guise of an EDSA exception. This means that the lack of precision in this category is not permitting violative content - rather, it points to the difficulty of distinguishing between TVEC0 (non-TVEC content) and TVEC.01 (content which would otherwise be TVEC but is permitted due to its educational, documentary, scientific, or artistic purpose). This could be improved through future policy refinement.

---
