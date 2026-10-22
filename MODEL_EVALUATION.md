# SMSense accuracy and robustness evaluation

## What this measures

SMSense combines two pretrained message classifiers with transparent behaviour, sender, URL and obfuscation evidence. Its 0–100 output is a **risk score, not a calibrated probability**. The evaluation runner measures the running API against labels you supply; it does not certify the detector or assume model-card results transfer to your messages.

The HinSpam model card documents label 0 as ham and label 1 as spam. SMSense now interprets that mapping explicitly when the standard HinSpam model is configured and uses semantic class names when available. The SilverGuard model card describes its ONNX output as an already-softmaxed scam probability in the 0–1 range. See the [HinSpam model card](https://huggingface.co/Keshav0av/HinSpam-spam-detection) and [SilverGuard model card](https://huggingface.co/tanishqmudaliar/SilverGuard). These are third-party model descriptions, not independent validation of SMSense on your dataset.

## Run an evaluation

1. Prepare artifacts/api-server/data/evaluation.csv using artifacts/api-server/data/evaluation_template.csv.
2. Label every row using a reviewer-verified ground truth. Supported benign labels are HAM, LEGITIMATE, BENIGN, SAFE, CLEAN, and NORMAL. Supported risky labels are SCAM, SPAM, PHISHING, FRAUD, and SMISHING.
3. Give each message a stable, non-sensitive source_id. Add language, scenario, and sender where known. Do not upload private SMS or real OTPs unless you have a lawful basis and have removed secrets.
4. Start the backend and wait until its required classifiers report ready.
5. From artifacts/api-server, run:

   ~~~powershell
   python tools/evaluate_dataset.py --csv data/evaluation.csv --base-url http://127.0.0.1:4444 --threshold 50 --report data/evaluation_report.json --predictions data/evaluation_predictions.csv
   ~~~

The runner uses /api/adversarial-test in batches of at most 20, so it does not create normal analysis-history entries. The prediction export omits message text by default. The report contains a combined-score confusion matrix and accuracy, precision, recall, specificity, F1, balanced accuracy and ROC-AUC, plus breakdowns by language and scenario. It also reports per-model score metrics when the API exposes them. Model-score comparisons are diagnostic; decide separate per-model thresholds on validation data before making a deployment decision.

## Build a test set that represents real use

Use separate, labelled groups instead of relying on one easy mixed sample:

- **Legitimate/ham:** ordinary personal messages, real bank debit/credit alerts, valid delivery updates, requested promotional SMS, real OTP notices, appointment reminders, and neutral messages that mention KYC/OTP/UPI without asking someone to disclose secrets.
- **Scams:** credential theft, fake KYC/account-block threats, UPI collect/payment fraud, lottery/reward fees, job-registration fees, fake parcel/customs charges, utility-disconnection threats, digital-arrest impersonation, malicious links, shortened links, and sender impersonation.
- **Language and form:** English, Hindi in Devanagari, Roman Hindi/Hinglish, code-switching, spelling mistakes, leetspeak, unusual Unicode, spaces inserted between letters, short messages, no-link scams, and messages with a sender ID as well as no sender.
- **Hard negatives and adversarial variants:** safety warnings such as “Never share your OTP”, a legitimate message with urgency, an otherwise innocent message containing a URL, and minimal edits that remove or introduce a request for payment or credentials.

Avoid leaking near-duplicate templates between training, validation and test sets. Where possible, separate by campaign/template family and time, not only by random row, so the final test contains genuinely unseen message patterns.

## Model selection: benchmark before switching

The current default combines HinSpam (designed for Hinglish spam/ham) with SilverGuard (an Indian SMS scam classifier). Their model cards document different training and input assumptions, so neither should be treated as universally best for every language or scenario. The SilverGuard card notes weaker reliability for very short messages and that it does not analyze embedded URLs; URL parsing and rules therefore remain separate evidence.

A multilingual IndicBERT-based scam classifier is another possible challenger for Hindi, Roman Hindi/Hinglish, and other Indian languages: [IndicBERT scam classifier v2](https://huggingface.co/anmolshrivastav/indicbert-scam-classifier-v2). Its model-card accuracy figures are author-reported results on a manually curated set; they are not independently verified here and must not be compared directly with HinSpam or SilverGuard scores. Do not add another always-on model just because its published headline metric is higher. First confirm its label mapping, license, dependencies, memory/latency, and performance on the same held-out SMSense test set, broken down by language and scam scenario.

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
