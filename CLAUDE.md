### Goal
Create an explainable AI dashboard suitable to be used by doctors and nurses for diagnosing pneumnoia, following the images provided in the design references (and their additional information in DESIGN_REFERENCES.md) and using the notes under "General Knowledge" for direction. Use benchmarking.pdf for more information about XAI in chestxray. Ask clarifying questions as needed.

### General knowledge
we’re researching Trustworthy & Explainable Medical AI (XAI): designed to understand how and why a model reached a conclusion. We need this to build trust in our results.

we don't want a black box - impossible to interpet, WHY:
Lots of math (trillions of interconnected parameters)
many dimensions, well above 3
this is also bad because it can disguise hidden biases
we want to understand the internal logic
If LIME already exists what else are we trying to do? Understand the whole model? What exactly is the goal of XAI - in technical detail. Because in a medical setting, can’t we literally just use a blackbox model to deliver a diagnosis, then provide a LIME explanation for each specific patient to understand how the model arrived at the conclusion?
*78% of doctors don’t trust AI they can’t understand, but 72% of doctors are using AI
Figure out how doctors are currently using AI, see what they are still missing, what they would like in terms of AI explanations
maybe talk to Rasha, ask her for more doctors to connect with
DM people on LinkedInn
Research what is already out there:
first figure out how doctors are already using AI (in diagnosis, what other usecases)
Figure out what issues they have with the technology
Trust?
How do they evaluate AI? How is Human-in-the-loop involved?


How doctors are using AI
Standout use cases:
Disease ID from patient data; real-time 3D spine views; epilepsy prediction from brain waves
Ambient AI scribes (e.g. 21% burnout drop at Mass General); ChatGPT for clinicians
Cardiac imaging/ECG-GPT (Yale CARS lab); drug discovery support; precision education
Prescription renewal (overstated—AI still hallucinates and isn't fully autonomous)
Where AI shines:
Pattern recognition & rules in data-rich fields (cardiology RCTs/EKG/imaging, oncology genomics/pathology, plastic surgery proportions)
Complex datasets (hematology, nephrology, infectious disease)—handles ~95% of cases
Limitations:
Incomplete information, social determinants, human complexity—not pure algorithms
Edge cases where protocols fail; not ready to replace clinical judgment
Key issues & risks:
If physicians don't adapt, tech firms may dictate care models and erode trust
Home personalized biomarker monitoring + recommendations
Psychiatry: AI sometimes perceived as more empathetic
EM: instant triage + drug interaction flags (won't replace final call or surgery)
Primary care: offloads repetitive tasks (med adherence, guideline updates) → more time for complex care
Messier/less profitable work stays with doctors
Liability for errors; regs (FDA ELSA review, SaMD, predetermined change control)
Bias/inadvertent redlining risk; prefer augment over replace




Papers with real visuals
The Importance of Contextual Information in Clinical Decision Support: A Neonatal-Sepsis Case Study
tests a dashboard for a neonatal late-onset sepsis (LOS) prediction model in the NICU, what patient context clinicians need to interpret and trust an ML risk score
makes hourly predictions from heart rate and oxygen saturation features
run only in shadow mode, so clinicians never saw its output & scores are also not calibrated, so they can’t be read as the probability of sepsis
What clinicians wanted:
To know what the prediction is based on, and how it has trended.
Interventions (ventilation, antibiotics), physical exam findings (skin color, activity), lab values (CRP, white cell count), and the patient’s weight and age.
All of it in one view, because the EHR scatters it across sources.
Risk shown as low, medium, or high categories, not numbers. Providers misread the raw score as the likelihood of sepsis.
Actionable guidance for each risk level, especially nurses (“do I call a doctor?”).
They disagreed on showing reliability metrics such as sensitivity and specificity, so the authors left them out.
Design decisions:
Patient overview on the left, details on the right.
A multi-line vital-signs chart with the last 6 hours.
A prediction trend chart that reveals feature contributions on hover or click, instead of a permanent feature-importance chart.
An alarm validation form asking whether the user agrees and what action they took.
Only red or gray tiles on the unit overview, because every alarm needs assessment.
Red and green coloring of abnormal values was removed because providers mistook it for the model’s drivers. One said the score was high because of skin color, which the model doesn’t use. Bold text replaced the color coding.

They treated the model as a second opinion or reassurance, not a decision-maker. In one case, a false-positive alarm made a provider look at the infant again.
They doubted a model that uses only heart rate and oxygen saturation, and found one prediction per hour too infrequent.
They wanted definitive actions, such as starting antibiotics for high risk. The model is only meant to flag elevated risk, so what they expected and what it’s for didn’t match.
Physician assistants and nurses scored higher than the neonatologist and fellow, who gave 60 and 70. same pattern in next study

Augmenting Clinical Decision-Making with an Interactive and Interpretable AI Copilot: A Real-World User Study with Clinicians in Nephrology and Obstetrics
Risk trajectory: a chart of predicted risk across visits, not a single score.
Risk factor list: features ranked by importance, with drill-down into each one’s trend.
LLM recommendation: a DeepSeek-generated narrative summary of the findings, with a prompt that forces it to stay grounded in the model’s evidence.
Population analysis: compares a patient’s indicator against cohort trends.

Users found the chart cluttered when many indicators overlapped.

Trust is built by verification. Clinicians tested the AI against their own reasoning, and being able to check its evidence raised their confidence even without better accuracy.
Experts and novices used it differently. Seniors hovered, paged, and cross-checked far more, which the authors call “adversarial verification.” Juniors used it as a “cognitive scaffold.”
Transparency cuts both ways. Implausible reasoning was spotted immediately and damaged trust. Examples include odd feature weights and an LLM citing “digestive system disease” when the value was zero.
Risk for juniors: The authors warn that the relief from information overload could encourage automation bias and de-skilling.


LIME
Treats the global model as a blackbox
Pick a single point
LIME creates a new dataset by making perturbations to the features of the original instance
Passes the newly generated samples through the original black-box model
Using a distance mtric (Euclidean distance, Gaussian kernel), LIME measures how far away each perturbed sample is from the original data point, assigns a higher weight to synthetic samples that are very close to the target instance and a lower weight to those that are further away
LIME fits a simple, glass-box model (sparse linear regression model/decision tree) on the weighted dataset of perturbed samples and their black-box predictions
Surrogate model is simple & local, directly inspect the parameters. These weights tell which specific features pushed the prediction up/down for the particular instance
The setup of XAI:
Technology requirements
prediction accuracy: comparing XAI output to results in training data set: most common technique: local interpertability model agnostic explanations (LIME): it explains the prediction of classifers by ML algorithm
trace ability: setups a narrower scope for ML rules & features, such as DeepLift (Deep learning important features): compares the activation of each neuron in the neural network to its reference neuron, showing traceability links and dependencies.
Human needs
decision understanding: could show the primary factors behind a decision, show the minimum changes required for the AI to produce a different outcome
building trust in the model & troubleshooting and building model performance: tracking model insights, deployment status, fairness quality, drift (degrade over time as production data differs from training data
model agnostic approaches: LIME, SHAP, PDPs, Ice plots, LAEs, an dfriedman's h-statistic plots (use with a level of caution, come with their own assumptions and limitations and only provide approximations for how the model makes predictions)
interpretability is a spectrum, not binary
at what point does a model go from being interpretable to explainable (grey area)
correlation matrices for linear regression
what does an effective explainable AI dashboard present clinicians with
perform research - but also a question to ask clinicians themselves
The prediction in context: the risk score or recommendation, with calibrated uncertainty or confidence and the clinical meaning of the number, not only the raw score.
Why the model said it: the top patient-specific drivers, such as labs, vitals, imaging regions or medications, with direction and size of effect. Use plain clinical language, not feature IDs.
What it can’t see: missing or stale inputs, out-of-distribution warnings, and known limits for this kind of patient. would be hard ?
Evidence for trust: how the model performs on similar patients and subgroups, and when it was last validated.
What could change the outcome: counterfactuals or “what-if” views, such as “if the creatinine were X, the risk would drop to Y.”
A way to act and push back: links to the underlying chart data, guideline context, and a one-click way to override or flag the output.
A fit with the workflow: progressive disclosure, so a glanceable summary comes first and detail is available on demand, with minimal added cognitive load.
Notes on existing implementations/trials
The Human Body is a Black Box”: Supporting Clinical Decision-Making with Deep Learning
Interpretability is deprioritized:
interpretability is not the only way, or even the main way, to earn trust and accountability in clinical AI
Sepsis has no agreed definition, and clinicians already use information they don’t fully understand, such as lab tests, and they don’t need causal explanations to act.
Method:
Risk scores are computed hourly for every adult ED patient. Patients above a 60% risk threshold show in red, and the positive predictive value at that threshold was 20%.
Sepsis criteria are checked every 5 minutes, and patients who meet them show in black.
Rapid response team (RRT) nurses watch the iPad app and phone the ED attending for every high-risk or septic patient. The physician makes the diagnosis, and the nurse cannot diagnose or treat.
Integrating a Machine Learning System Into Clinical Workflows: Qualitative Study
Implementation
Dashboard: Nurses said the tablet app was easy to use. The color-coded cards (black for meets criteria, red high, orange medium, yellow low) let them see risk at a glance, and the three tabs made tracking simple. Nurses still opened the EHR chart to confirm what they saw.
Human communication: Physicians said a phone call from a nurse worked better than EHR pop-up alerts, which they tend to ignore. One compared the nurses to air traffic controllers.
Nurse strategies: Nurses grouped patients by ED area into one call, avoided calling at shift change, asked if it was a good time, and cited chart details to show they had done their homework.
Barriers:
Physicians were sometimes busy or unreachable, and calls could take an hour or two to land.
Calls interrupted workflow, especially when they added nothing new.
Nurses and physicians didn’t know each other, and nurses who monitor remotely felt that made escalation harder.
Physicians initially didn’t understand the program, and some feared malpractice liability.
Research suggests Epic Sepsis Model is lacking in predictive power
The authors also raised concerns about the model being proprietary and opaque. They urged professional organizations to make formal recommendations about these algorithms.
shows alert fatigue is a real risk for broadly deployed models, including at high alert volumes (Duke paper chooses to resolve by using the RRT nurse as a human intermediary)
thresholds must be set per user type echoes Duke’s choice to tune thresholds with local clinicians (the 60% cutoff, 20% PPV)
Duke team also argued for local validation, as opposed to trusting vendor-reported performance

Let’s try building our own dashboard? Maybe based on the saliency papers - pretending to be a clinician? Maybe we use GradCAM and all the other saliency methods and display them?

I’ll claude code this up

Yes it should be technical - but in real use, this is going to be clinician-facing, so it should be explainable in their terms: what factors (like a decision tree, SBP, age, etc.) are leading AI to these conclusions

I’ll use the following as design references.. named by:
TICICDS = The Importance of Contextual Information in Clinical Decision Support and ACDIIAIC = Augmenting Clinical Decision-Making with an Interactive and Interpretable AI Copilot
Ex. TICICDS-fig1.png