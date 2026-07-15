# Operationalizing Conformal Anomaly Detection: Detection, Calibration, Error Control, and Deployment Beyond Idealized Conditions

## Abstract

Deployed anomaly detection systems rarely provide statistically interpretable expressions of uncertainty, making it difficult to trust their outputs in high-stakes operational environments. Existing approaches to uncertainty quantification often rely on heuristics, asymptotics, or parametric assumptions and typically lack a direct statistical guarantee connecting the reported uncertainty to the reliability of the resulting decisions in practice. This leaves a gap between uncertainty quantification and decision-level error control, especially when false alarms carry operational costs.


This dissertation studies conformal inference as a framework for closing this gap. In the form of conformal anomaly detection, the framework is non-parametric and model-agnostic, turning anomaly scores into calibrated measures of statistical evidence, such as conformal $p$- or $e$-values, with finite-sample validity under data exchangeability. The resulting measures connect upstream anomaly scoring to downstream statistical error-control procedures within an integrated inference pipeline.

Building on this pipeline view, this dissertation addresses the theory-practice gap in conformal anomaly detection, moving beyond statements of statistical validity to optimize operational utility. The theoretical analysis characterizes fundamental limitations of standard conformal anomaly detection within its intended settings, alongside those of existing extensions beyond exchangeability. It demonstrates that even when finite-sample validity is formally preserved, the resulting measures of evidence can become uninformative under operational realities---particularly in regimes defined by limited data, distribution shift, or dependence. In response, the work develops targeted adaptations and new methods at the intersection of anomaly scoring, calibration, and error-control layers, with sequential inference and decision-making under evolving data serving as central stress tests. The dissertation thereby contributes towards an operational theory and methodology of conformal anomaly detection in which validity, statistical power, robustness, and deployment constraints are managed as coupled design objectives.

## Preamble

Machine learning systems are increasingly used to support decision-making across individual, organizational, societal, and policy-level domains. As their outputs inform consequential decisions, it becomes essential to quantify uncertainty in a way that is both statistically interpretable and operationally useful. While many approaches to uncertainty quantification exist in machine learning, they often rely on assumptions, or heuristic uncertainty proxies. These restrictions can become problematic in deployment settings where assumptions are violated, or uncertainty estimates do not translate into calibrated evidence or actionable and reliable decision rules.

A particularly relevant framework in this context is conformal inference. Compared to many other approaches to uncertainty quantification, conformal methods offer a strong balance between flexibility and statistical rigour. In their standard form, they are post-hoc applicable and provide distribution-free, finite-sample guarantees under data exchangeability. These properties make conformal inference a principled frequentist complement to Bayesian uncertainty quantification: Bayesian methods provide coherent probabilistic uncertainty conditional on a specified prior–likelihood model, whereas conformal methods calibrate model outputs using data to provide finite-sample reliability guarantees under exchangeability.

Applied to anomaly detection, conformal inference provides a natural framework for calibrating anomaly scores. In the standard construction, a test point's score is compared with the scores of a calibration sample, yielding a conformal $p$-value through its relative rank. Under exchangeability of the calibration and null test observations, these $p$-values are finite-sample valid without distributional assumptions and enable principled statistical error control.

A growing body of literature studies conformal inference beyond the exchangeable setting. However, translating these advances to the specific requirements of anomaly detection remains an active and challenging research problem. The unsupervised nature of anomaly scoring introduces unique difficulties, necessitating further methodological developments to balance statistical validity and operational utility---particularly in non-standard, dynamic and operationally constrained settings.

## Objectives and Research Questions

The objective of this dissertation is to study when and how the statistical guarantees of conformal anomaly detection (\textbf{CAD}) translate into practically useful decisions---and when they do not. On this basis, the dissertation develops new methods, and adapts existing ones, to better align CAD with the statistical requirements \textit{and} operational constraints of anomaly detection in real-world deployment settings.

CAD is an assumption-light frequentist framework. It does not require a correctly specified probabilistic model, asymptotic approximations, or parametric assumptions to obtain finite-sample validity under exchangeability. This robustness comes with a trade-off: the guarantees concern statistical validity, not necessarily informativeness, statistical power, or operational usefulness. In particular, CAD can remain formally valid while becoming too conservative to support any discoveries.

To bridge this gap, it is necessary to identify exactly where the disconnect between validity and utility can emerge:

### Ineffective anomaly scoring

If the underlying anomaly detector does not separate normal from anomalous observations well, the resulting anomaly scores carry little useful information. This does not invalidate conformal calibration: conformal $p$- or $e$-values can still satisfy their formal validity guarantees under the relevant assumptions. However, the resulting statistical evidence will be weak. Downstream error-control procedures may then have little statistical power, because the scoring layer provides insufficient signal.

### Restrictive conformal calibration

Even when anomaly scores are informative, the calibration step may impose fundamental limitations. Small calibration sets restrict the lower bound of conformal $p$-values, while extensions beyond exchangeability often preserve validity only at the cost of additional conservativeness, stronger assumptions, or estimation error. Such regimes need not invalidate finite-sample guarantees, but may inflate conformal $p$-values. As a result, the calibrated evidence may become too conservative to support reliable detection.

### Stringent error control

The last source of restrictions arises at the decision layer. Strict error control requirements, low anomaly prevalence, or sequential decision-making can cause control procedures to become overly conservative. Again, this does not undermine validity. Rather, stringent error control may lead to few or no discoveries, with weak evidence further amplifying this effect. In such cases, the system preserves its guarantees at the expense of practical detection utility.

Accordingly, improving CAD requires anomaly-scoring methods that remain informative under the limited feedback of genuinely unsupervised settings, conformal calibration procedures that translate heuristic scores into statistically principled evidence while expressing certainty when warranted, and error-control methods whose assumptions and target metrics reflect the requirements of practical anomaly-detection tasks.

To achieve this, the research is guided by three core objectives:

#### Theoretical Analysis

Systematically evaluate CAD as an integrated inference pipeline to characterize the systemic failure modes that emerge in practical applications. Specifically, this objective aims to describe how specific limitations and inefficiencies interact and compound to eventually compromise operational utility and statistical power, despite maintaining formal statistical guarantees.

#### Methodological Advancement

Develop and empirically evaluate targeted improvements for the most restrictive bottlenecks in the CAD pipeline, with a practical focus on sequential decision-making. Depending on the failure modes identified, this includes improving anomaly scoring, adapting calibration procedures, or adjusting error control and corresponding metrics to better align with operational requirements.

#### Deployment-Oriented Guidance

Distill these theoretical, methodological, and empirical findings into actionable guidance for selecting, adapting and deploying CAD methods under specific operational constraints.

In synthesis, these objectives comprehensively span problem identification, targeted methodological enhancement of critical deployment bottlenecks, and the translation of theoretical and empirical insights into actionable guidelines for real-world impact.

To achieve these objectives, the dissertation will systematically address the following interconnected research questions:

{\textbf{``How do limitations within the integrated CAD pipeline interact and compound to produce systemic operational failure modes despite preserving formal theoretical validity?''}
Specifically, what structural dependencies or data characteristics accelerate this transition from formal validity to operational informativeness collapse, even in extensions that seek to adapt to them?

\textbf{``How can targeted methodological enhancements to restrictive pipeline bottlenecks improve the operational power and robustness of CAD, particularly within sequential decision-making environments?''}
Under what conditions must given validity guarantees be reformulated into localized or context-specific guarantees to ensure the system remains operationally viable without losing explicit statistical safeguarding?

\textbf{``What structural principles and decision frameworks allow the translation of theoretical limits and empirical or methodological insights into actionable guidance for selecting, adapting and deploying CAD configurations under known operational constraints?''}
How can evaluating metrics be dynamically balanced when operational definitions of ``useful detection behaviour'' conflict with classic statistical optimization criteria?

## Scientific Relevance

This dissertation addresses a critical gap at the intersection of statistical learning theory and applied machine learning. While conformal inference has seen rapid theoretical advancement, the literature remains overwhelmingly focused on supervised learning tasks where ground-truth feedback is readily available. By contrast, CAD operates in a fundamentally unsupervised and \textit{ill-posed} setting. Advancing the conformal framework within this domain requires moving beyond standard assumptions to address the unique challenges of anomaly detection.

Scientifically, this work shifts the focus from pure statistical validity to operational utility. Most advancements in conformal inference are deeply theoretically grounded which not always translates into practicality and operational relevance. By formalizing the interactions between anomaly scoring, conformal calibration, and downstream error control, the dissertation provides a much-needed theoretical framework for evaluating conformalized anomaly detection workflows and informing improvements and adaptions or fundamental operational limitations. Characterizing the failure modes of CAD contributes directly to the broader statistical discourse on robust uncertainty quantification.

Conformal inference, as the underlying framework, is scientifically relevant in the field of uncertainty quantification because it provides a model-agnostic and distribution-free framework for deriving finite-sample guarantees from calibration data. Rather than requiring a specific predictive model, it operates through user-defined conformity or nonconformity scores, such as residuals, reconstruction errors, or anomaly scores. Its validity does not depend on correct model specification, but on exchangeability between the calibration data and the test data under the null hypothesis. Under this assumption, conformal methods provide statistically interpretable uncertainty measures without parametric distributional assumptions, although the informativeness of these measures still depends on the quality of the underlying scores and the amount of calibration data available.

## Contribution

This dissertation aims to make methodological, theoretical, and empirical contributions, with sequential inference tasks as a primary application context:

### Theoretical Contributions

This dissertation characterizes the operational limits of CAD as an integrated inferential pipeline. It studies when standard finite-sample validity guarantees translate into statistically actionable evidence, when valid conformal outputs become too weak or conservative to support useful decisions, and how these limitations arise across the scoring, calibration, and error-control layers. Particular attention is given to practically relevant regimes such as limited data, dependence, and distribution shift where existing conformal extensions may preserve formal validity while still failing to deliver operational utility.


### Methodological Contributions

This dissertation develops targeted methodological advances for the main bottlenecks of the CAD pipeline: anomaly scoring, conformal calibration, and downstream statistical error control. Rather than treating these layers independently, the work studies how methodological choices at one layer affect the informativeness, robustness, and decision usefulness of the full system. The resulting methods are designed for data-scarce, dynamic, and sequential anomaly-detection settings, with the aim of improving operational power while retaining explicit statistical safeguards.

### Empirical and Deployment-Oriented Contributions

The dissertation provides an empirical basis for evaluating CAD under realistic operating conditions. Through controlled experiments, simulations and representative case studies, it quantifies trade-offs between validity, power, robustness, error-control stringency, computational cost, and interpretability. These analyses inform practical guidance for selecting, adapting, and deploying CAD methods under known constraints, and clarify when methodological improvements materially change detection performance rather than only preserving formal guarantees. 