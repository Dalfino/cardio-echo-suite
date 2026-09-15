# Data Use Agreement Template — cardio-echo-suite

> **Instructions**: This DUA governs the sharing of de-identified clinical data between a hospital (the "Provider") and the cardio-echo-suite development team (the "Recipient"). Customize all [BRACKETS]. Have institutional counsel review before signing.

---

## Data Use Agreement

This Data Use Agreement ("DUA") is entered into between **[INSTITUTION NAME]**, a [STATE] [non-profit/corporation] with offices at [ADDRESS] (the "Provider"), and **[RECIPIENT NAME]**, a [STATE] [non-profit/corporation] with offices at [ADDRESS] (the "Recipient").

### 1. Background

The Provider maintains clinical databases of echocardiogram and electrocardiogram (ECG) studies performed at [INSTITUTION NAME]. The Recipient has developed an open-source software package called "cardio-echo-suite" that uses artificial intelligence to interpret echocardiograms and ECGs. The parties wish to collaborate on a research study to validate the accuracy of cardio-echo-suite on the Provider's de-identified clinical data.

### 2. Definitions

- **"De-identified Data"** means clinical data that has been de-identified in accordance with the HIPAA Safe Harbor method described in 45 CFR 164.514(b), such that no remaining information could be used to identify an individual patient.
- **"Derived Results"** means the output of the cardio-echo-suite software when run on the De-identified Data, including AI predictions, FHIR R4 resources, audit logs, and statistical analyses.
- **"Software"** means the cardio-echo-suite computer code, including all source code, model weights, documentation, and related materials, available at https://github.com/[RECIPIENT GITHUB]/cardio-echo-suite.

### 3. Data Sharing

#### 3.1 Provider Responsibilities

The Provider agrees to:
- (a) Identify a sample of 300-500 echocardiogram studies and 300-500 ECG studies performed at [INSTITUTION NAME] between [START DATE] and [END DATE].
- (b) De-identify all studies per HIPAA Safe Harbor before transfer to the Recipient.
- (c) Provide cardiologist-labeled ground truth for each study (EF, rhythm, valve findings) extracted from final signed reports.
- (d) Provide demographic metadata (age, sex, BMI, race/ethnicity, scanner vendor) in de-identified form.
- (e) Transfer the De-identified Data to the Recipient via secure encrypted transfer (e.g., SFTP with PGP encryption, or HIPAA-compliant cloud storage).

#### 3.2 Recipient Responsibilities

The Recipient agrees to:
- (a) Run the cardio-echo-suite Software on the De-identified Data.
- (b) Generate Derived Results including per-study AI predictions and aggregate statistics.
- (c) Share the Derived Results with the Provider for review and quality improvement.
- (d) Not attempt to re-identify any individual patient from the De-identified Data.
- (e) Not redistribute the De-identified Data to any third party without prior written consent from the Provider.
- (f) Store the De-identified Data on secured, access-controlled servers located at [SPECIFY LOCATION].
- (g) Destroy the De-identified Data within 90 days of study completion, and provide written certification of destruction.

### 4. Permitted Uses

The Recipient may use the De-identified Data and Derived Results solely for:
- (a) Validating the accuracy of cardio-echo-suite
- (b) Publishing aggregate results in peer-reviewed scientific journals
- (c) Improving the cardio-echo-suite Software
- (d) Preparing FDA 510(k) or De Novo submissions for cardio-echo-suite, where [INSTITUTION NAME] is acknowledged as a validation site

### 5. Prohibited Uses

The Recipient may NOT:
- (a) Use the De-identified Data for any commercial product without prior written consent
- (b) Attempt to re-identify any individual patient
- (c) Redistribute the De-identified Data to third parties
- (d) Use the De-identified Data for purposes other than those listed in Section 4
- (e) Use the De-identified Data to develop a competing product without [INSTITUTION NAME]'s consent

### 6. Intellectual Property

- (a) The Provider retains all rights, title, and interest in the De-identified Data.
- (b) The Recipient retains all rights, title, and interest in the cardio-echo-suite Software, which is licensed under the MIT License (see https://github.com/[RECIPIENT GITHUB]/cardio-echo-suite/blob/main/LICENSE).
- (c) The Derived Results are jointly owned by the Provider and Recipient. Either party may use the Derived Results for research, publication, or regulatory submission purposes, with the other party acknowledged.
- (d) Model weights from upstream academic projects (EchoNet-Dynamic, PanEcho, ECG-FM, MedSAM2, nnU-Net) retain their original upstream licenses. Neither party may sub-license these weights.

### 7. Publication

- (a) Either party may publish aggregate results from this study in peer-reviewed journals or at scientific conferences.
- (b) The publishing party will provide the other party with a draft of any manuscript at least 30 days before submission for review and comment.
- (c) The Provider may delay publication for up to 60 days to protect intellectual property, but may not unreasonably withhold publication.
- (d) Both parties will be acknowledged in any publication, with authorship determined by scientific contribution.

### 8. Confidentiality

Both parties agree to keep confidential any non-public information shared during this collaboration, including but not limited to:
- (a) The terms of this DUA
- (b) Any identified vulnerabilities in the Software
- (c) Any [INSTITUTION NAME] internal policies or procedures learned during the collaboration

Confidentiality obligations survive termination of this DUA for a period of 5 years.

### 9. Term and Termination

- (a) This DUA is effective as of the date of last signature and remains in effect for the duration of the study, plus 5 years for data retention.
- (b) Either party may terminate this DUA with 30 days written notice.
- (c) Upon termination, the Recipient will destroy all De-identified Data within 90 days and provide written certification of destruction.

### 10. HIPAA Compliance

Both parties acknowledge that the De-identified Data is not Protected Health Information (PHI) under HIPAA because it has been de-identified per 45 CFR 164.514(b). However, both parties will handle the De-identified Data with reasonable security measures to prevent re-identification.

### 11. Indemnification

Each party will be responsible for its own acts and omissions. Neither party will be liable to the other for indirect, incidental, or consequential damages arising from this collaboration.

### 12. Governing Law

This DUA is governed by the laws of the State of [STATE], without regard to conflict of law principles.

### 13. Entire Agreement

This DUA constitutes the entire agreement between the parties regarding the subject matter and supersedes all prior agreements.

### 14. Signatures

**[INSTITUTION NAME]**

By: ________________________________

Name: [NAME]

Title: [TITLE]

Date: ____________

**[RECIPIENT NAME]**

By: ________________________________

Name: [NAME]

Title: [TITLE]

Date: ____________

---

*This template is provided as a starting point only. Both parties should have institutional counsel review and customize this DUA before signing.*
