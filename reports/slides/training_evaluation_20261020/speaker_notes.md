# Four-model training and evaluation

## 01. Four models, every completed learning stage

This is a dedicated training/evaluation presentation rather than the full project-history deck. The requested four families are included; MiniCPM is outside this deck. Six CLIP visual runs, three SmolVLM LoRA runs and the corrected supervised VadCLIP study are documented. Later classifier, region and source-data studies are separated from genuine changes to pretrained backbone weights. No model is retrained for this documentation build and no speed or latency measurements are included.

## 02. The stages differ by model architecture

MobileCLIP updates 18,106,992 visual parameters plus 10,245 head parameters; SigLIP updates 14,176,512 plus 15,365. Earlier visual layers and text encoders remain frozen. SmolVLM updates 4,341,760 language projection parameters while its vision tower and connector remain frozen. VadCLIP trains about 12.642 million adapter parameters over frozen CLIP-B/16 features. A missing SmolVLM linear probe is not assigned another model's result.

## 03. Smoking: all four model families, stage by stage

MobileCLIP/SigLIP stages are the original CLIP visual study: 3,258 train,1,633 dev. SmolVLM original and joint LoRA use the expanded 3,973 train/1,812 dev corpus and matched binary prompting. They share the646-case test with the earlier CLIP stages but not the same training data. VadCLIP stages share their own1,056-frame test and9/4/3session split. Its temporal adapter, Smol language LoRA and CLIP visual fine-tunes are distinct forms of weight adaptation. SmolVLM has no executed linear-probe stage. This AP table must not be read as a matched four-model leaderboard.

## 04. Overall data coverage, without double counting studies

The expansion adds 1,118 smoking photos: 715 training, 179 development and 224 test. The main SmolVLM corpus therefore has 3,973 training, 1,812 development, 2,025 test and 806 external examples. Additional transfer is wholly held out from that training. HMDB later changes from a 906-clip transfer cohort to grouped adaptation folds; ten title-part aliases are purged, leaving 896. The 3,278 VadCLIP frames reuse the original CCTV corpus under a different split. Views, repeated temporal endpoints and copied dataset exports do not create independent events.

## 05. What the five targets are actually trained on

Source counts in the primary CLIP manifest: CCTV smoking 3,278; Stanford40 1,406; IWDD 435; Street Scene 496; apparent public sleep 113; unattended endpoints 1,770. Subsequent photo expansion adds 1,118. IWDD acquired 498/500 videos; two missing training videos and 63 training videos linked to official test sources are excluded, leaving 274/61/100. Old person-only loitering exports, duplicate lying/standing images and foreground-mask datasets are not treated as complete target supervision. Public-sleep labels describe appearance/context, with no inference about housing status.

## 06. Primary CLIP dataset: total / positive by target

Smoking trains on camera c4 and Stanford40 official-training photos, develops on c2 and tests c1; the external 806 photos preserve Stanford40 official-test membership. IWDD retains its 100 official test clips. Street Scene recordings stay within a split, but all originate from one camera. Sleep splits use author/perceptual groups. Unattended training has two source groups, development one and historical test two camera families. Static targets use one endpoint, temporal tasks four chronological views, and littering eight uniform clip views.

## 07. Expanded strategy and SmolVLM dataset: total / positive

The original CLIP visual fine-tuning study and this expanded corpus do not have identical training data. The 16 October CLIP strategy routes and 19 October SmolVLM study do share these expanded training/development splits and exactly matched held-out IDs/labels. That is the basis for the later three-model pipeline comparison. Main held-out cases total 2,831; additional transfer adds 961, giving 3,792. Neither count equals independent people, cameras or events.

## 08. VadCLIP uses a separate session-held-out smoking split

The split is 9 training, 4 development and 3 test sessions, grouped by camera/date. The 1,056 test frames contain 560 positives and 496 normals. Train and test both include camera IDs c1, c2 and c4 on different dates, so this is session generalization rather than unseen-camera generalization. The separate camera-c1 holdout used by the five-target CLIP/Smol studies has 646 frames and 393 positives. No arithmetic comparison between those scores is a matched four-model benchmark.

## 09. Transfer cohorts answer different generalization questions

HMDB transfer uses eight uniform views across full clips and includes 109 smoking plus 797 primary-action controls. The later adapted pilot uses sixteen views and 530/161/205 grouped clips after title-part alias purging. MEVA has 15 abandonment windows plus 12 pre-event controls across ten synchronized sessions/seven cameras, using source end annotations to define offline windows. It does not supply fully reviewed unattended-owner state. XRF55 train/dev/test aliases and scene blocks are disjoint; training inspection did not verify a visible cigarette, so it is weak gesture evidence.

## 10. Train, select and test play different roles

Historical CCTV, unattended intervals and many transfer cohorts were inspected in earlier rounds; those tests remain retrospective. Exact pixel disjointness does not establish complete semantic/source independence. Six CLIP visual runs use seeds 42/43/44; Smol uses two primary learning rates at seed 42 plus one repeat at seed 43. More source views, repeated photos or adjacent endpoints do not increase the count of independent target events. The XRF55 split was frozen before new source predictions. No latency experiment is included.

## 11. Use ranking and decision metrics together

Main detailed stage tables use each method's maximum-development-F1 cutoff, frozen before held-out scoring. Supplementary tables explicitly use raw zero or the development 5% FPR point. Older CLIP and Smol cutoffs use pooled target development negatives; later 17/20 October smoking experiments constrain every development source. Accuracy and balanced accuracy are derived from saved TP/FP/FN/TN. All source thresholds and unrounded metrics are included in metrics.csv. AP/AUPR labels follow the stored implementations; VadCLIP AP belongs to its separate frame cohort. Endpoint FPR is not continuous false alerts per camera-hour.

## 12. Linear probing and visual tuning change different weights

The probe pools ordered input image embeddings into normalized endpoint/mean/max/last-minus-first features. C is selected on development AP: primary MobileCLIP C10, SigLIP C100. Only one annotated target contributes BCE to visual tuning. Fine-tuning starts from the selected probe head, updates the recorded last-stage visual parameters, and applies feature distillation plus weight anchoring at coefficient0.1 each. Scene-preserving augmentation uses horizontal flip0.5, brightness/contrast0.15 and saturation0.1, without random crops. Float32 parameters with BF16 autocast train on GPU; final selection/calibration and held-out inference use float32 with TF32 disabled. Zero-shot uses a text direction on the endpoint embedding rather than the supervised temporal pooling, so that transition changes supervision and representation.

## 13. Six actual visual fine-tuning runs were completed

Both primary models select visual learning rate3e-5 from1e-5/3e-5. Epochs maximize development macro AP, averaging source AP within each of the five targets. Checkpoints and per-task routes are frozen before final test. Original AMP diagnostics are preserved; a numerical audit found batch-shape sensitivity, so the final tables use the separate float32_evaluation outputs. Operating thresholds were recalibrated only on float32 development predictions, keeping weights and method selection unchanged.

## 14. MobileCLIP: every training stage on the same holdout

Zero-shot text uses the endpoint visual embedding; the probe and visual-tuned classifier use normalized last/mean/max/change features. Thus the first step adds supervision and representation pooling. Visual tuning versus the probe uses the same representation and dataset. These bars are all candidates, not a test-best deployment selection. AP does not determine a working cutoff. Values come from the final float32 evaluation, not the retained AMP diagnostic.

## 15. SigLIP2: every training stage on the same holdout

Zero-shot text uses the endpoint visual embedding; the probe and visual-tuned classifier use normalized last/mean/max/change features. Thus the first step adds supervision and representation pooling. Visual tuning versus the probe uses the same representation and dataset. These bars are all candidates, not a test-best deployment selection. AP does not determine a working cutoff. Values come from the final float32 evaluation, not the retained AMP diagnostic.

## 16. Smoking / CCTV: zero-shot → probe → visual tune

Primary seed 42. AP and ROC-AUC are ranking metrics independent of the displayed cutoff. F1, precision, recall and FPR use each method's own frozen maximum-development-F1 threshold. Zero-shot/probe/visual-tune stages use the same held-out labels; the zero-shot endpoint and supervised pooling representations differ. Dataset/source qualifications apply: historical CCTV and unattended labels are retrospective; IWDD is whole-clip waste dumping; loitering is one-camera supervised windows; sleep is18 provisional photos with6 positives; Stanford40 is an official photo holdout from a domain contributing training examples. Test-best method values do not override development routing.

## 17. Littering / IWDD: zero-shot → probe → visual tune

Primary seed 42. AP and ROC-AUC are ranking metrics independent of the displayed cutoff. F1, precision, recall and FPR use each method's own frozen maximum-development-F1 threshold. Zero-shot/probe/visual-tune stages use the same held-out labels; the zero-shot endpoint and supervised pooling representations differ. Dataset/source qualifications apply: historical CCTV and unattended labels are retrospective; IWDD is whole-clip waste dumping; loitering is one-camera supervised windows; sleep is18 provisional photos with6 positives; Stanford40 is an official photo holdout from a domain contributing training examples. Test-best method values do not override development routing.

## 18. Loitering: zero-shot → probe → visual tune

Primary seed 42. AP and ROC-AUC are ranking metrics independent of the displayed cutoff. F1, precision, recall and FPR use each method's own frozen maximum-development-F1 threshold. Zero-shot/probe/visual-tune stages use the same held-out labels; the zero-shot endpoint and supervised pooling representations differ. Dataset/source qualifications apply: historical CCTV and unattended labels are retrospective; IWDD is whole-clip waste dumping; loitering is one-camera supervised windows; sleep is18 provisional photos with6 positives; Stanford40 is an official photo holdout from a domain contributing training examples. Test-best method values do not override development routing.

## 19. Public sleep: zero-shot → probe → visual tune

Primary seed 42. AP and ROC-AUC are ranking metrics independent of the displayed cutoff. F1, precision, recall and FPR use each method's own frozen maximum-development-F1 threshold. Zero-shot/probe/visual-tune stages use the same held-out labels; the zero-shot endpoint and supervised pooling representations differ. Dataset/source qualifications apply: historical CCTV and unattended labels are retrospective; IWDD is whole-clip waste dumping; loitering is one-camera supervised windows; sleep is18 provisional photos with6 positives; Stanford40 is an official photo holdout from a domain contributing training examples. Test-best method values do not override development routing.

## 20. Unattended: zero-shot → probe → visual tune

Primary seed 42. AP and ROC-AUC are ranking metrics independent of the displayed cutoff. F1, precision, recall and FPR use each method's own frozen maximum-development-F1 threshold. Zero-shot/probe/visual-tune stages use the same held-out labels; the zero-shot endpoint and supervised pooling representations differ. Dataset/source qualifications apply: historical CCTV and unattended labels are retrospective; IWDD is whole-clip waste dumping; loitering is one-camera supervised windows; sleep is18 provisional photos with6 positives; Stanford40 is an official photo holdout from a domain contributing training examples. Test-best method values do not override development routing.

## 21. Smoking / Stanford40 photos: zero-shot → probe → visual tune

Primary seed 42. AP and ROC-AUC are ranking metrics independent of the displayed cutoff. F1, precision, recall and FPR use each method's own frozen maximum-development-F1 threshold. Zero-shot/probe/visual-tune stages use the same held-out labels; the zero-shot endpoint and supervised pooling representations differ. Dataset/source qualifications apply: historical CCTV and unattended labels are retrospective; IWDD is whole-clip waste dumping; loitering is one-camera supervised windows; sleep is18 provisional photos with6 positives; Stanford40 is an official photo holdout from a domain contributing training examples. Test-best method values do not override development routing.

## 22. The chosen route is not always the most fine-tuned model

Primary MobileCLIP routes all five targets through its fine-tuned checkpoint. Primary SigLIP routes smoking/unattended through visual tuning, dumping/public sleep through its probe and loitering through zero-shot. SigLIP's sleep fine-tune has a higher test AP than the selected probe; that fact is retained rather than retrospectively replacing the development route. These scores remain on the original 7,498-example study; later expanded-data strategies are a separate comparison.

## 23. Unattended items: the cutoff changes the conclusion

No test threshold is fitted. The raw-zero cutoff was predeclared, so reporting it alongside development-max-F1 and development-budget results is a sensitivity analysis rather than threshold cherry-picking. AP is unchanged across all three rows for a route. Normal endpoint FPR remains different from false alerts per continuous camera-hour. Item ownership and temporal consistency are not measured by high endpoint AP.

## 24. Keep seed 42 primary and report the other seeds

The range includes seeds42/43/44; each replication follows its own development-only routing and the primary selected visual LR. Seeds do not add independent cameras, people, new labels or events. The full CSV contains every method, seed and threshold. No winning test seed replaces seed 42. Small public-sleep and loitering cohorts make apparent score changes especially unstable.

## 25. After backbone training: regions, geometry and new heads

The expanded training/development pool is3,973/1,812. Automatic Faster R-CNN contextual crops have confidence 0.5,20% padding and full-frame fallback. Full pooled features use last/mean/max/change; ordered features concatenate four indexed frames plus mean/max. MLP uses one64-unit ReLU layer, dropout 0.3, AdamW LR 0.001/weight decay 0.01, batch 128, at most80 epochs and patience 8. Linear classifiers use LBFGS up to400 iterations. Visual interpolation uses only previously trained parameters. Primary selection maximizes mean development-source AP. Secondary worst-source analyses and later policies have their own timing qualifications.

## 26. Five-target comparison on identical held-out cases

Existing provenance checked exact held-out IDs/labels for all3,792 cases. The new presentation also asserts all cohort denominators and class counts match across the three saved pipelines. Primary development-selected CLIP strategy routes are compared with primary seed 42 SmolVLM per-target routes. The SmolVLM original control remains the causal comparison for LoRA weight adaptation. No VadCLIP score is inserted because its separate1,056-frame smoking test has different sessions. Later HMDB-adapted205-clip results use additional training and do not replace the906-clip transfer row. These historical cohorts remain retrospective.

## 27. Photo and action-transfer comparison on matched cases

Existing provenance checked exact held-out IDs/labels for all3,792 cases. The new presentation also asserts all cohort denominators and class counts match across the three saved pipelines. Primary development-selected CLIP strategy routes are compared with primary seed 42 SmolVLM per-target routes. The SmolVLM original control remains the causal comparison for LoRA weight adaptation. No VadCLIP score is inserted because its separate1,056-frame smoking test has different sessions. Later HMDB-adapted205-clip results use additional training and do not replace the906-clip transfer row. These historical cohorts remain retrospective.

## 28. Matched cohorts: ranking gains versus operating errors

Smoking cohort646/393 positive; unattended962/242 positive. Method-specific development-budget thresholds are fixed. These older studies constrain pooled development negatives for each target, rather than every source separately. The table includes AP, ROC-AUC, F1, precision, recall and FPR on exactly matched held-out labels. No low-FPR operating threshold is fitted to these test cases. Confusion counts and thresholds are retained in metrics.csv; precision and F1 can look favorable with high positive prevalence even when normal rejection fails.

## 29. LoRA adapts the answer function while vision stays frozen

A sample contributes supervision only for its annotated target. The Yes/No first-token loss uses the whole vocabulary, with all prompt/image loss positions excluded. Vision tower, multimodal connector, token embeddings and vocabulary head are immutable. Rank 8alpha16dropout0.05 target seven projection types in language layers. Linear LR decay follows5% warmup; gradient norm clipping1.0. One epoch has7,946 visits: all3,973unique examples plus an equally sized task/class/source/group-balanced resample. Frozen vision tokens are512-pixel inputs without image splitting; static tasks1image, temporal windows4, littering8. Held-out logits are float32 with TF32 off. Original and adapted routes use identical prompts and images.

## 30. All three adapter runs select epoch one

Development metric averages dataset-source AP within each target, then equally across five targets. Original matched development macro AP0.5657; primary joint adapter seed 42/LR 1e-4 reaches0.7026. The low-rate candidate reaches0.6584; repeated seed 43/LR 1e-4 reaches0.6548. Each run completes three epochs under patience 2 and selects epoch 1. Epoch loss is not directly comparable to CLIP's binary classifier loss. Per-task routing chooses between original and the two primary adapters using development data before held-out inference; it is not a newly trained third head.

## 31. Original model → one joint LoRA → per-target routing

Original is the unadapted model under the same binary query, not the earlier six-label JSON generation baseline. Joint uses high-rate seed 42adapter for all targets; primary routing selects low-rate seed 42 for public sleep and high-rate for the other four. Loitering original AP0.4250 falls to0.1822 despite favorable primary development selection. Public sleep joint0.6327 versus routed0.7885 demonstrates why a joint model and a routed system must not be conflated. This table uses exact stored AP values rather than generated class labels.

## 32. SmolVLM: full decision metrics after adaptation

All six rows use the frozen maximum-development-F1 operating point. The source-specific positive/negative counts are in the dataset and exported metrics tables. This binary task query has no trained unknown category, explanations or boxes. It is not numerically comparable to the older six-class free-generation accuracy. Native image forwards validate the cached visual-token path. Low-FPR and raw-zero operating points follow on the next slide; they do not use test fitting.

## 33. SmolVLM: high endpoint recall can mean many false positives

At the pooled-development-budget cutoff, unattended recall94.6%, FPR70.8%. Smoking CCTV recall1.8%, FPR0%. At raw zero, unattended recall64.9%, FPR25.8%; CCTV predicts no positives. These outcomes illustrate different score distributions and source offsets. Test cutoffs are unchanged. All three points share the same AP/AUC because only the threshold varies. Neither static-frame accuracy nor sparse-window AP measures persistent owner departure or long normal camera exposure.

## 34. A second seed exposes unstable task gains

The second seed is an optimization repeat, not an independent data-source replication. Seed 42joint sleep AP0.6327, seed 43joint0.9167 and primary routed low-rate0.7885 are different configurations. Optional repeated-gain routing was formulated after primary held-out inspection; it is exploratory and not substituted into these primary tables. Exact original baseline rows repeat across seeds only as controls, not as new model runs.

## 35. VadCLIP learns a sequence model on frozen CLIP-B/16

The pretrained CLIP-B/16 feature/text encoder stays frozen. The baseline logistic regression uses C1, max_iter2000 and training-frame labels, without test tuning. The adapter uses masked per-frame binary visual loss plus alignment cross entropy. Padded frames do not contribute. It can use context inside an offline sequence; raw-frame scores are expanded from the representative sequence to original labels. Best checkpoint/branch uses development PR performance, selecting epoch 1; training stops after51completed epochs under patience 50. Test threshold comes only from312development frames. This session split shares physical cameras across dates and differs from the other models' unseen-camera holdout. Historical earlier adapter ablations with no final test are excluded.

## 36. VadCLIP: zero-shot → probe → temporal adapter

The text baseline AP0.4823, linear probe0.5439, dual-mean adapter0.6332 all use identical test labels. The adapter's ROC-AUC0.5874, F10.5648, precision0.6210, recall0.5179, FPR0.3569, accuracy0.5767. Exact confusion:290TP,177FP,270FN,319TN. The probe's development cutoff flags97.8% of test normals, despite its high recall. Always-positive F1≈0.693 exceeds adapter F1 while producing100% FPR, demonstrating why F1 alone is misleading. AP/AUPR are taken from the saved implementation; this is not a matched-cohort comparison with646-case camera-c1 tests.

## 37. Four adapted models: metrics with their protocol labels

A includes three pipelines sharing held-out camera-c1 frames but different primary training corpora and score interfaces. MobileCLIP/SigLIP entries are the actual visual-tune candidate rather than a later strategy selected after source-data expansion. SmolVLM is the high-rate primary joint LoRA. B is the corrected separate VadCLIP temporal adapter test. Do not infer a winner by sorting AP or F1 across A/B; prevalence is60.8% versus53.0%, and split difficulty differs. The preceding within-study stage tables are the valid attribution comparisons. The expanded-data matched three-model table uses a different, explicitly named strategy stage.

## 38. Later learning adds automatic regions and temporal memory

Each encoder fits14smoking and11unattended heads. Person/region baselines, causal GRUs with and without geometry, three seeds and region-supervised variants are retained. Hard negatives are mined only from training normals beginning epoch 4; a checkpoint selected earlier did not benefit from later mining. Smoking train3,342/dev1,475 combines expanded static data and HMDB adaptation; unattended uses236/236eligible windows. Ten known film-part aliases were removed from training. Improved crop geometry does not prove improved anomaly classification, and sparse ownership geometry remains a hypothesis. This training stage differs from both the earlier partial visual fine-tunes and SmolVLM language LoRA.

## 39. Later smoking stages: CCTV frames, n646

Scene probe and region/person rows use their original pooled development-budget cutoff. The17October policy chooses a fixed ensemble and single global cutoff that satisfies each development source's5% FPR budget. Primary policy uses person-supervised hard-negative seed 42 for MobileCLIP and all-head ensemble for SigLIP. These policies are selected on development data, while historical test sources were already inspected. The205HMDB test has30 positive/175controls, additional530training clips and16 views; the earlier906clip Smol/strategy transfer cohort had no HMDB training and8 views. Do not compare those APs as one fixed-test learning curve.

## 40. Later smoking stages: Adapted HMDB test, n205

Scene probe and region/person rows use their original pooled development-budget cutoff. The17October policy chooses a fixed ensemble and single global cutoff that satisfies each development source's5% FPR budget. Primary policy uses person-supervised hard-negative seed 42 for MobileCLIP and all-head ensemble for SigLIP. These policies are selected on development data, while historical test sources were already inspected. The205HMDB test has30 positive/175controls, additional530training clips and16 views; the earlier906clip Smol/strategy transfer cohort had no HMDB training and8 views. Do not compare those APs as one fixed-test learning curve.

## 41. 20 October: old-only, new-only and mixed training

XRF55 official public RGB release v6 supplies740selected clips and11,840 frames. Training-only inspection found masked actors performing hand-to-mouth gestures; class15 is retained as a publisher action proxy, not verified target smoking. New-only fitting uses340weakly labelled clips, not a verified CCTV smoking corpus. Frozen pretrained encoders and exact automatic contextual crops feed logistic heads with C0.1/1/10. Each source/class/group is balanced. All36fits converge; both model development selections freeze before test. No new temporal rule, visual fine-tune or language adapter is trained. Old-only also receives new-source development calibration, and new-only receives old-source development feedback.

## 42. Data-addition outcome: Historical CCTV

Mixed SigLIP CCTV AP0.7309to0.6227, and FPR3.95% to36.76%. On the new action proxy AP0.3568to0.7768, with recall21.7% and FPR1.7% at the frozen cutoff. MobileCLIP also loses CCTV AP0.6937to0.5673; its mixed proxy cutoff finds no positives and flags20drinking clips. All source control errors in these CLIP families occur on drinking actions. The new-source subject-level recall varies strongly; mixed SigLIP recalls10%,0%,55% across the three aliases. HistoricalCCTV remains retrospective, and proxy gain does not establish true CCTV improvement.

## 43. Data-addition outcome: New action proxy

Mixed SigLIP CCTV AP0.7309to0.6227, and FPR3.95% to36.76%. On the new action proxy AP0.3568to0.7768, with recall21.7% and FPR1.7% at the frozen cutoff. MobileCLIP also loses CCTV AP0.6937to0.5673; its mixed proxy cutoff finds no positives and flags20drinking clips. All source control errors in these CLIP families occur on drinking actions. The new-source subject-level recall varies strongly; mixed SigLIP recalls10%,0%,55% across the three aliases. HistoricalCCTV remains retrospective, and proxy gain does not establish true CCTV improvement.

## 44. New-source SmolVLM transfer after prior LoRA

Both routes use the same8uniform frames selected from the16-view source manifest; the existing native prompt supports at most8. Each route scores160 dev clips and freezes new-source thresholds before either scores test. Original new-dev-budget recall61.7%, FPR36.7%; LoRA35.0%,18.3%. Older19Octoberbudget cutoffs yield zero recall and zero FPR on both routes. No scene4 adapter selection or new LoRA training occurs. Native image/cache margin checks pass. The source remains scripted-gesture evidence and cannot certify real smoking performance.

## 45. What the four-model evidence supports for each target

The most defensible comparisons are within-study baseline/probe/adaptation stages and expanded-data matched three-pipeline cohorts. Generalization is still limited by camera diversity, independent annotation and confusable negatives. Apparent public sleep remains a visual proxy, and static endpoints do not verify owner departure or excessive dwell duration. The separate VadCLIP study supports only smoking-session results. Full-parameter model unfreezing was not performed. No test-best seed, retrospective cutoff or optional repeated-gain routing is silently promoted. The new deck reports existing training/evaluation facts without another inference or latency experiment.

## 46. Detailed numbers and training choices are available to review

Companion directory: reports/slides/training_evaluation_20261020. The CSV exports stored AP, ROC-AUC, precision, recall, F1, FPR, TP/FP/FN/TN, as well as derived accuracy/balanced accuracy. The manifest binds every reporting input and builder. A separate validation checks source hashes, metrics against the original result cells, PowerPoint/PDF content and rendered page bounds. The8-page20Octoberreport and102-page19Octobertechnical history remain separate documents. This deck focuses on the four requested model families and their training/evaluation stages.
