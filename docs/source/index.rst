Nearby documentation
====================

Nearby is a Python package that provides metrics to quantify the variability and stability of neural responses in EEG.
It integrates various variability measures within a unified API and a consistent processing workflow.
The package supports time-frequency representations such as ERD/S as well as time-domain signals including ERP data.

In the fields of BCI, it is well known that individual differences and performance variability 
cannot be fully explained by traditional measures such as average ERD/S or average ERP.
In particular, motor-imagery BCI studies report that even with sufficient training,
approximately 15-30% of users fail to achieve satisfactory performance.
Recently, neural response variability has been highlighted as a potential factor contributing to this issue.
Nearby was developed to address these research needs and to provide a reproducible and
user-friendly tool for studying neural variability.

Using Nearby, researchers can easily perform analyses such as:

- Investigating neural variability as a factor explaining inter- and intra-subject performance differences in BCI
- Evaluating trial-to-trial and within-trial stability of ERP and ERD/S responses
  
Citations
=========

If you use Nearby in your study, please cite the following publication.

- Simon Kojima, Fabien Lotte, "Quantifying Event-Related (De)Synchronization Variability for Brain-Computer Interface: A Unified and Interpretable Framework", arXiv preprint arXiv:2607.21119, `https://arxiv.org/abs/2607.21119 <https://arxiv.org/abs/2607.21119>`_

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   documentation
   auto_examples/index
   install