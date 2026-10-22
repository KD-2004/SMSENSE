# SMSense model selection and accuracy evaluation

## Current model decision

The primary detector is **SMS-Shield 4B Q4_K_M**, run locally through Ollama:

- [SMS-Shield model card](https://huggingface.co/ganmoor-ai-labs/sms-shield)
- [OpenTextShield alternative](https://huggingface.co/telecomsxchange/OpenTextShield)
- [IndicBERT scam classifier v2](https://huggingface.co/anmolshrivastav/indicbert-scam-classifier-v2)

SMS-Shield is the closest fit found for this project because the published model contract explicitly targets Indian SMS, English, Hindi/Devanagari and romanized Hinglish; accepts sender/header context; and returns a structured verdict, category, scam type, risk band and red flags. Its quantized Q4_K_M file is about 2.4 GB according to its model card, and it can run locally using Ollama. The project has been updated to use this model as the primary scorer instead of requiring HinSpam plus SilverGuard.

The model card reports 99.2% verdict accuracy, 99.5% dangerous recall and no genuine OTP/bank-alert false flags on its 1,390-example held-out set. **These are the model author's reported numbers, not independently verified SMSense results.** The card also states that its 12.5k-message training corpus is synthetic and that distribution shift is expected. It does not establish that the same metrics will hold on real SMS. We therefore use its score as one capped evidence source alongside URL, behaviour, sender and obfuscation checks; the score is not a calibrated probability.

I compared it with OpenTextShield, an MIT-licensed multilingual BERT SMS classifier with a smaller runtime footprint. Its own model card is usefully candid about modern phishing: it reports 72.4% combined block rate and 45.7% phishing recall on its independent 8,007-message IMC 2025 benchmark, while noting that other benchmark sets overlap its training data. That makes it an interesting challenger, but not the best first choice for SMSense's Indian scam focus. IndicBERT's reported 99.64% is from a small manually curated set of 1,400 and is also author-reported, so it remains an optional diagnostic challenger instead of being added to the production ensemble.

**Why not combine every model?** More models can add correlated errors, memory use and latency. This change selects one task-aligned primary model and keeps the fusion transparent. The next decision should come from an independently labelled SMSense benchmark, not headline percentages from different model cards.

The main model card only specifies 512 characters per request. SMSense therefore classifies longer pasted texts in overlapping 512-character segments and takes the highest segment risk. That is a deliberate recall-first policy; it can increase false positives on long copied conversations and must be measured.

## Run an evaluation

1. From the artifacts/api-server directory, prepare a starter dataset from any CSV that has a message column:

   ~~~powershell
   python tools/prepare_evaluation_csv.py --input "C:\path\to\your_messages.csv" --output data\evaluation.csv
   ~~~

   The helper preserves message text and copies source_id, sender, language, and scenario when present. It leaves label blank by default and never infers ground truth from a model prediction. You must fill the label column after independent review. Use --verified-label-column only when the source column contains independently human-verified labels.
2. Use one of the supported labels for each message. Benign labels: HAM, LEGITIMATE, BENIGN, SAFE, CLEAN, NORMAL. Scam labels: SCAM, SPAM, PHISHING, FRAUD, SMISHING. Include at least one message from both classes. Do not label a message from SMSense's prediction alone.
3. Give each message a stable, non-sensitive source_id. Add language, scenario, and sender where known. Do not upload private SMS or real OTPs unless you have a lawful basis and have removed secrets.
4. Start the backend and wait until the API is ready.
5. From artifacts/api-server, run:

   ~~~powershell
   python tools/evaluate_dataset.py --csv data\evaluation.csv --base-url http://127.0.0.1:4444 --threshold 50 --report data\evaluation_report.json --predictions data\evaluation_predictions.csv
   ~~~

The evaluator rejects a missing file, missing/blank/unsupported labels, duplicate source IDs, and a dataset that contains only one class. It calls /api/adversarial-test in batches of at most 20, so it does not create normal analysis-history entries. The prediction export omits message text by default. The report contains a combined-score confusion matrix and accuracy, precision, recall, specificity, F1, balanced accuracy and ROC-AUC, plus breakdowns by language and scenario. It also reports per-model score metrics when the API exposes them. Model-score comparisons are diagnostic; decide separate per-model thresholds on validation data before making a deployment decision.

If you do not yet have verified labels, you can still use tools/run_sample_and_report.py for operational diagnostics; those results do not establish accuracy.

## Build a test set that represents real use

Use separate, labelled groups instead of relying on one easy mixed sample:

- **Legitimate/ham:** ordinary personal messages, real bank debit/credit alerts, valid delivery updates, requested promotional SMS, real OTP notices, appointment reminders, and neutral messages that mention KYC/OTP/UPI without asking someone to disclose secrets.
- **Scams:** credential theft, fake KYC/account-block threats, UPI collect/payment fraud, lottery/reward fees, job-registration fees, fake parcel/customs charges, utility-disconnection threats, digital-arrest impersonation, malicious links, shortened links, and sender impersonation.
- **Language and form:** English, Hindi in Devanagari, Roman Hindi/Hinglish, code-switching, spelling mistakes, leetspeak, unusual Unicode, spaces inserted between letters, short messages, no-link scams, and messages with a sender ID as well as no sender.
- **Hard negatives and adversarial variants:** safety warnings such as “Never share your OTP”, a legitimate message with urgency, an otherwise innocent message containing a URL, and minimal edits that remove or introduce a request for payment or credentials.

Avoid leaking near-duplicate templates between training, validation and test sets. Where possible, separate by campaign/template family and time, not only by random row, so the final test contains genuinely unseen message patterns.

## Model selection and next challenger

SMS-Shield is the primary model for Indian SMS classification. Its model card reports results on its own 1,390-example held-out test set and notes that its training corpus is synthetic. We have not independently verified those results on real messages; model-card scores should be treated as a candidate signal, not a production accuracy guarantee.

IndicBERT remains an optional diagnostic challenger for Hindi, Roman Hindi/Hinglish and other Indian-language cases. Enable it only after installing \`requirements-optional-ml.txt\` and setting \`INDICBERT_ENABLED=true\`. It reports a separate model score and does not influence the fused production score. Its model-card figures are author-reported from a manually curated test set and are not directly comparable with SMS-Shield's results.

OpenTextShield is a useful future candidate if broader multilingual support becomes a priority. Its card reports a 72.4% combined block rate and 45.7% phishing recall on an 8,007-message modern multilingual benchmark, and acknowledges that two other benchmarks overlap its training data. Compare it with SMS-Shield using exactly the same independent SMSense dataset before considering a switch. See [OpenTextShield](https://huggingface.co/telecomsxchange/OpenTextShield).

## Threshold and model decisions

- Pick the score threshold using a **validation set**. Then freeze it and report final results once on an untouched test set. Do not choose a threshold from the final test results.
- Review false negatives and false positives separately. For scam detection, a high recall with too many false alarms may be unusable, while a high precision with missed scams may be unsafe.
- Check per-language and per-scenario coverage; one aggregate score can hide weak performance on Roman Hindi, no-link scams, or legitimate transactional SMS.
- Do not infer that a score of 80 means an 80% probability of a scam. The fused score has not been calibrated against your labelled data.
- Do not change score weights merely to improve the apparent result on the same data used to assess them. Tune on validation, then re-run on a separate test set.
- Keep unknown model label mappings as an error rather than guessing. If you replace a model through environment variables, confirm its label semantics and evaluate it before relying on its output.

NIST guidance similarly recommends realistic test sets, clearly documented methods, measurement of both false positives and false negatives, stress testing, and monitoring performance across relevant conditions: [NIST AI RMF Measure guidance](https://airc.nist.gov/airmf-resources/playbook/measure/).

## Current validation status

This repository provides regression tests and a repeatable evaluation tool. Until the tests have passed in CI and SMSense has been measured against your independently labelled corpus, do not advertise a numeric accuracy claim.
