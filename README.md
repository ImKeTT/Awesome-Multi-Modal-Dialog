# Awesome Paper List of Multi-Modal Dialogue



![logo](./logo.png)



Papers, codes and resources about multi-modal dialogue, including methods, datasets and related metrics.

We split the multi-modal dialogue task to **Visual-Grounded Dialogue (VGD, including Visual QA or VQA), Visual Question Generation, Multimodal Conversation (MMC)** and **Visual Navigation (VN)**.

## Datasets

|                           Dataset                            |                Task                |  Publisher  |       Author        |
| :----------------------------------------------------------: | :--------------------------------: | :---------: | :-----------------: |
|  [VQA:  Visual Question Answering](http://cloudcv.org/vqa)   |          visual QA (VQA)           |  ICCV 2015  |    Virginia Tech    |
|       [Visual  Dialog](https://visualdialog.org/data)        |          visual QA (VQA)           |  CVPR 2017  |  VisualDialog Org.  |
| [GuessWhat?!  Visual object discovery through multi-modal  dialogue](https://guesswhat.ai/download) |          visual QA (VQA)           |  CVPR 2017  |   Montreal Univ.    |
| [Visual  Reference Resolution using Attention Memory for Visual  Dialog](http://cvlab.postech.ac.kr/research/attmem) |          visual QA (VQA)           |  NIPS 2017  |   Postech&Disney    |
| [CLEVR:  A diagnostic dataset for compositional language and elementary visual  reasoning](https://cs.stanford.edu/people/jcjohns/clevr/) |          visual QA (VQA)           |  CVPR 2017  |      Stanford       |
| [Image-grounded  conversations: Multimodal context for natural question and response  generation (IGC)](https://www.microsoft.com/en-us/download/details.aspx?id=55324) |          visual QA (VQA)           | IJCNLP 2017 | Rochester&Microsoft |
| [Towards  Building Large Scale Multimodal Domain-Aware Conversation Systems  (MMD)](https://amritasaha1812.github.io/MMD/) |       multimodal conv. (MMC)       |  AAAI 2018  |         IBM         |
| [Embodied Question Answering (EQA)](https://embodiedqa.org/) |          visual QA (VQA)           |  CVPR 2018  |      Facebook       |
| [Talk the walk: Navigating new york city through grounded  dialogue](https://github.com/facebookresearch/talkthewalk) |       visual navigation (VN)       |  ICLR 2019  |        MILA         |
|      [Vision-and-Dialog  Navigation](https://cvdn.dev/)      |       visual navigation (VN)       |  CoRL 2019  |         UoW         |
| [CLEVR-Dialog:  A Diagnostic Dataset for Multi-Round Reasoning in Visual  Dialog](https://github.com/satwikkottur/clevr-dialog) |    visual-grounded dialog (VGD)    | NAACL 2019  |         CMU         |
| [Image-Chat:  Engaging Grounded Conversations](http://parl.ai/projects/image_chat) |    visual-grounded dialog (VGD)    |   ACL2020   |      Facebook       |
|    [OpenViDial](https://github.com/ShannonAI/OpenViDial)     |    visual-grounded dialog (VGD)    | arxiv 2020  |      ShannonAI      |
| [Situated  and Interactive Multimodal Conversations  (SIMMC)](https://github.com/facebookresearch/simmc) | multimodal conv./visual navigation | COLING 2020 |      Facebook       |
| [PhotoChat:  A human-human dialogue dataset with photo sharing behavior for joint  image-text  modeling](https://github.com/google-research/google-research/tree/master/multimodalchat/) |      multimodal  conv. (MMC)       |  ACL 2021   |       Google        |
| [MMConv:  An Environment for Multimodal Conversational Search across Multiple  Domains](https://github.com/lizi-git/MMConv) |       multimodal conv. (MMC)       | SIGIR 2021  |         NUS         |
| [Constructing  Multi-Modal Dialogue Dataset by Replacing Text with Semantically Relevant  Images](https://github.com/shh1574/multi-modal-dialogue-dataset) |       multimodal conv. (MMC)       |   ACL2021   |        KAIST        |
| [OpenViDial  2.0: A Larger-Scale, Open-Domain Dialogue Generation Dataset with Visual  Contexts](https://github.com/ShannonAI/OpenViDial) |    visual-grounded dialog (VGD)    | arxiv 2021  |      ShannonAI      |
| [MMChat:  Multi-Modal Chat Dataset on Social Media](https://github.com/silverriver/MMChat) |    visual-grounded dialog (VGD)    |  LREC 2022  |       Alibaba       |
| [MSCTD:  A Multimodal Sentiment Chat Translation  Dataset](https://github.com/XL2248/MSCTD) |    visual-grounded dialog (VGD)    | arxiv 2022  |       Tencent       |



## Methods

We roughly split the learning paradigm of different methods (if available) as: *Fusion-Based (FB)* and *Attention-Based (AB)*.

- *Fusion-based (FB)*: Simple concatenation of multi-modal information at the model input.
- *Attention-Based (AB)*: Co-attention between different modalities to learn their relations.

### Visual Grounded Dialogue

Visual grounded dialogue considers only **one image** for one dialogue session. The whole session is constrained to this given image. It is also know as *Visual Dialog* task.

|                            Title                             |                         Dataset Used                         |  Publisher   |                             Code                             | Class |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------: | :----------------------------------------------------------: | :---: |
|      [Visual  Dialog](https://arxiv.org/abs/1611.08669)      |                         VisDial v0.9                         |  ICCV 2017   |                             CODE                             |  FB   |
| [Open Domain Dialogue Generation with Latent  Images](https://arxiv.org/abs/2004.01981) |                      Image-Chat; Reddit                      |  AAAI 2021   |                             CODE                             |  FB   |
| [Maria:  A Visual Experience Powered Conversational  Agent](https://arxiv.org/abs/2105.13073) |                  Reddit; Conceptual Caption                  |   ACL 2021   |         [CODE](https://github.com/jokieleung/Maria)          |  FB   |
| [Learning  to Ground Visual Objects for Visual Dialog](https://arxiv.org/abs/2109.06013) |                 VisDial v0.9, v1.0;  MS-COCO                 |   ACL 2021   |                             CODE                             |  FB   |
| [Multi-Modal  Open-Domain Dialogue](https://arxiv.org/abs/2010.01082) | Image-Chat; ConvAI2;  EmpatheticDialogues; Wizard of WikiPedia; BlendedSkillTalk |  EMNLP 2021  |                             CODE                             |  FB   |
| [Iterative  Context-Aware Graph Inference for Visual  Dialog](https://arxiv.org/abs/2004.02194) |                      VisDial v0.9, v1.0                      |  CVPR 2020   |        [CODE](https://github.com/wh0330/CAG_VisDial)         |  FB   |
| [Reasoning  Visual Dialog with Sparse Graph Learning and Knowledge  Transfer](https://arxiv.org/abs/2004.06698) |                         VisDial v1.0                         |  EMNLP 2021  |     [CODE](https://github.com/gicheonkang/SGLKT-VisDial)     |  FB   |
| [VD-BERT:  A Unified Vision and Dialog Transformer with  BERT](https://arxiv.org/abs/2004.13278) |                      VisDial v0.9, v1.0                      |  EMNLP 2020  |        [CODE](https://github.com/salesforce/VD-BERT)         |  FB   |
| [GuessWhat?!  Visual object discovery through multi-modal  dialogue](https://arxiv.org/abs/1611.08481) |                    Guessing;MNIST Dialog                     |  CVPR 2017   |      [CODE](https://github.com/GuessWhatGame/guesswhat)      |  FB   |
| [Ask  No More: Deciding when to guess in referential visual  dialogue](https://arxiv.org/abs/1805.06960) |                           Guessing                           | COLING 2018  |          [CODE](https://vista-unitn-uva.github.io)           |  FB   |
| [Visual  Reference Resolution using Attention Memory for Visual  Dialog](https://papers.nips.cc/paper/2017/hash/654ad60ebd1ae29cedc37da04b6b0672-Abstract.html) |                 MNIST Dialog; VisDial  v1.0                  |  NIPS 2017   |                             CODE                             |  AB   |
| [Visual  Coreference Resolution in Visual Dialog using Neural Module  Networks](https://arxiv.org/abs/1809.01816) |                    MNIST Dialog; VisDial                     |  ECCV 2018   |     [CODE](https://github.com/facebookresearch/corefnmn)     |  AB   |
| [Dual  Attention Networks for Visual Reference Resolution in Visual  Dialog](https://arxiv.org/abs/1902.09368) |                      VisDial v1.0, v0.9                      |  EMNLP 2019  |      [CODE](https://github.com/gicheonkang/DAN-VisDial)      |  AB   |
| [Efficient  Attention Mechanism for Visual Dialog that can Handle All the Interactions  between Multiple Inputs (LTMI)](https://arxiv.org/abs/1911.11390) |                         VisDial v1.0                         |  ECCV 2020   |         [CODE](https://github.com/davidnvq/visdial)          |  AB   |
| [Large-scale  Pretraining for Visual Dialog: A Simple State-of-the-Art  Baseline](https://arxiv.org/abs/1912.02379) | Wikipedia; BooksCorpus;  Conceptual Cations; VQA; VisDial v1.0 |  ECCV 2019   |      [CODE](https://github.com/vmurahari3/visdial-bert)      |  AB   |
| [Multi-step  Reasoning via Recurrent Dual Attention for Visual  Dialog](https://arxiv.org/abs/1902.00579) |                    VisDial v1.0; MS-COCO                     |   ACL 2019   |                             CODE                             |  AB   |
| [Are  You Talking to Me? Reasoned Visual Dialog Generation through Adversarial  Learning](https://arxiv.org/abs/1711.07613) |                         VisDial v0.9                         |  CVPR 2018   |                             CODE                             |  AB   |
| [Best  of Both Worlds: Transferring Knowledge from Discriminative Learning to a  Generative Visual Dialog Model](https://arxiv.org/abs/1706.01554) |                         VisDial v0.9                         | NeurIPS 2017 |     [CODE](https://github.com/jiasenlu/visDial.pytorch)      |  AB   |
| [Multi-View  Attention Network for Visual Dialog](https://arxiv.org/abs/2004.14025) |                      VisDial v1.0, v0.9                      |  arxiv 2020  | [CODE](https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch) |  AB   |
| [Learning  Cooperative Visual Dialog Agents with Deep Reinforcement  Learning](https://arxiv.org/abs/1703.06585) |                           VisDial                            |  ICCV 2017   |     [CODE](https://github.com/batra-mlp-lab/visdial-rl)      |  FB   |
| [Multimodal  Hierarchical Reinforcement Learning Policy for Task-Oriented Visual  Dialog](https://arxiv.org/abs/1805.03257) |                           VisDial                            | SIGDIAL 2018 |                             CODE                             |  FB   |
| [Beyond  task success: A closer look at jointly learning to see, ask, and  GuessWhat](https://arxiv.org/abs/1809.03408) |                        GuessingWhat?!                        |  NAACL 2019  |          [CODE](https://vista-unitn-uva.github.io)           |  FB   |

### Multi-modal Conversation

Multi-modal conversation (MMC) aims at conducting conversations with multiple images. Models should understand multiple images and/or generate multi-modal responses during conversation.

|                            Title                             |                         Dataset Used                         | Publisher  |                             Code                             | Class |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :--------: | :----------------------------------------------------------: | :---: |
| [Multimodal  Dialogue Response Generation](https://arxiv.org/abs/2110.08515) |                 PhotoChat; Reddit; YFCC100M                  |  ACL 2022  |                             CODE                             |  FB   |
| [Constructing  Multi-Modal Dialogue Dataset by Replacing Text with Semantically Relevant  Images](https://arxiv.org/abs/2107.08685) | DailyDialog; Empathetic Dialog; Persona Chat; MS-COCO; Flicker30K |  ACL 2021  | [CODE](https://github.com/shh1574/multi-modal-dialogue-dataset) |  FB   |
| [Towards  Enriching Responses with Crowd-sourced Knowledge for Task-oriented  Dialogue](https://dl.acm.org/doi/10.1145/3475959.3485392) |                           MMConv.                            | MuCAI 2021 |                             CODE                             |  FB   |
| [Multimodal  Dialog System: Generating Responses via Adaptive  Decoders](https://liqiangnie.github.io/paper/fp349-nieAemb.pdf) |                             MMD                              |  MM 2019   |       [CODE](https://acmmultimedia.wixsite.com/magic)        |  FB   |
| [Multimodal  Dialog Systems via Capturing Context-aware Dependencies of Semantic  Elements](http://staff.ustc.edu.cn/~tongxu/Papers/Weidong_MM20.pdf) |                             MMD                              |  MM 2020   |          [CODE](ttps://github.com/githwd2016/MATE)           |  FB   |
| [Multimodal  Dialog System: Relational Graph-based Context-aware Question  Understanding](https://dl.acm.org/doi/abs/10.1145/3474085.3475234) |                             MMD                              |  MM 2021   |                             CODE                             |  FB   |
| [User  Attention-guided Multimodal Dialog  Systems](https://dl.acm.org/doi/10.1145/3331184.3331226) |                             MMD                              | SIGIR 2019 |          [CODE](  https://github.com/ChenTsuei/UMD)          |  FB   |
| [Text  is NOT Enough: Integrating Visual Impressions into Open-domain Dialogue  Generation](https://arxiv.org/abs/2109.05778) |             DailyDialog; Flickr30K;  PersonaChat             |  MM 2021   |                             CODE                             |  AB   |

### Question Generation

Question generation task generates questions instead of responses based on given images. It is similar to visual-grounded dialogue.

|                            Title                             |     Dataset Used     | Publisher |                         Code                         |
| :----------------------------------------------------------: | :------------------: | :-------: | :--------------------------------------------------: |
| [Category-Based  Strategy-Driven Question Generator for Visual  Dialogue](https://aclanthology.org/2021.ccl-1.89.pdf) |    GuessingWhat?!    | CCL 2021  |                         CODE                         |
| [Visual  Dialogue State Tracking for Question  Generation](https://arxiv.org/abs/1911.07928) |    GuessingWhat?!    | AAAI 2020 |                         CODE                         |
| [Answer-Driven  Visual State Estimator for Goal-Oriented Visual  Dialogue](https://dl.acm.org/doi/10.1145/3394171.3413668) | GuessWhat?!; MS-COCO |  MM 2020  | [CODE](https://github.com/zipengxuc/ADVSE-GuessWhat) |
| [Goal-Oriented  Visual Question Generation via Intermediate Re-  wards](https://openaccess.thecvf.com/content_ECCV_2018/papers/Junjie_Zhang_Goal-Oriented_Visual_Question_ECCV_2018_paper.pdf) |     GuessWhat?!      | ECCV 2018 |                         CODE                         |
| [Learning  goal-oriented visual dialog via tempered policy  gradient](https://arxiv.org/abs/1807.00737) |     GuessWhat?!      | SLT 2018  |                         CODE                         |
| [Information  maximizing visual question generation](https://arxiv.org/abs/1903.11207) |         VQG          | ICCV 2019 |                         CODE                         |

### Visual Navigation

Visual navigation focuses on guiding users to their destination from their starter points given surrounding information. Here we mainly collect methods that involve conversational guidance in language.

|                            Title                             |                         Dataset Used                         | Publisher  |                             Code                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :--------: | :----------------------------------------------------------: |
| [Learning  to interpret natural language navigation instructions from observations](http://www.cs.utexas.edu/~ai-lab/pubs/chen.aaai11.pdf) | [Walk the Talk](https://www.aaai.org/Papers/AAAI/2006/AAAI06-232.pdf) | AAAI 2011  |                             CODE                             |
| [Talk the walk: Navigating new york city through grounded  dialogue](https://arxiv.org/abs/1807.03367) |                  Talk the Walk; GuessWhat?!                  | Arxiv 2018 |   [CODE](https://github.com/facebookresearch/talkthewalk)    |
| [Mapping  Navigation Instructions to Continuous Control Actions with  Position-Visitation Prediction](https://arxiv.org/abs/1811.04179) |           [LANI](https://arxiv.org/abs/1809.00786)           | CoRL 2018  |           [CODE](https://github.com/lil-lab/drif)            |
| [Vision-and-Language  Navigation: Interpreting visually-grounded navigation instructions in real  environments](https://arxiv.org/abs/1711.07280) | [R2R](https://github.com/google-research/google-research/tree/master/r4r); VQA; [Matterport3D](https://arxiv.org/abs/1709.06158) | CVPR 2018  | [CODE](https://github.com/peteanderson80/Matterport3DSimulator) |
| [Embodied Question Answering](https://arxiv.org/abs/1711.11543) |                           EQA; VQA                           | CVPR 2018  |    [CODE](https://github.com/facebookresearch/EmbodiedQA)    |
| [IQA:  Visual Question Answering in Interactive  Environments](https://arxiv.org/abs/1712.03316) |                     IQUAD; VQA; AI2-THOR                     | CVPR 2018  | [CODE](https://github.com/danielgordon10/thor-iqa-cvpr-2018) |
| [Natural  language navigation and spatial reasoning in visual street  environments](https://arxiv.org/abs/1811.12354) |  Touchdown; VQA; ReferItGame; Google  Refexp; Talk the Walk  | CVPR 2019  |         [CODE](https://github.com/lil-lab/touchdown)         |
| [Stay on the Path: Instruction Fidelity in Vision-and-Language  Navigation](https://arxiv.org/abs/1905.12255) | [R2R; R4R](https://github.com/google-research/google-research/tree/master/r4r) |  ACL 2019  |                             CODE                             |
| [Learning  to navigate unseen environments: Back translation with environmental  dropout](http://aclanthology.lst.uni-saarland.de/N19-1268.pdf) |                         Matterport3D                         | NAACL 2019 |       [CODE](https://github.com/airsplay/R2R-EnvDrop)        |
| [Touchdown: Natural language navigation and spatial reasoning in visual street  environments](https://arxiv.org/abs/1811.12354) |  Touchdown; VQA; ReferItGame; Google  Refexp; Talk the Walk  | CVPR 2019  |         [CODE](https://github.com/lil-lab/touchdown)         |



## Metrics

A summary paper of visual dialogue metrics: [A Revised Generative Evaluation of Visual Dialogue](https://arxiv.org/abs/2004.09272), [CODE](https://github.com/danielamassiceti/geneval_visdial).

We split the related metrics to *Rank-based* and *Generiate-based*

- *Rank-based*: measures the quality of responses that **retrieve** from response candidates.
- *Generate-based*: measures the quality of responses that **generate** from the model.

#### Rank-based Metrics

|                       Metrics                       | Better indicator |                         Explanation                          |
| :-------------------------------------------------: | :--------------: | :----------------------------------------------------------: |
|                        Mean                         |   $\downarrow$   |       mean rank of ground truth response in candidates       |
|                        R@k                        |    $\uparrow$    | ratio of ground truth response in the top-k ranked responses |
|             Mean Reciprocal Rank (MRR)              |    $\uparrow$    | mean reciprocal rank of the ground truth response in the ranked responses |
| Normalized Discounted Cumulative Gain@k(NDCG@k) |    $\uparrow$    | relevance score list, assigns 0-1 for 100 candidates responses based on semantic similarity with the ground truth responses |

#### Generate-based Metrics

|       Metrics       | Better indicator |
| :-----------------: | :--------------: |
|      BLEU-k       |    $\uparrow$    |
|    ROUGE-k/L    |    $\uparrow$    |
|       Meteor        |    $\uparrow$    |
|        CIDEr        |    $\uparrow$    |
| Embedding *Average* |    $\uparrow$    |
| Embedding *Extrema* |    $\uparrow$    |
| Embedding *Greedy*  |    $\uparrow$    |

#### Model-based Metrics
The model-based metrics are evaluated by off-the-shelf language models or multimodal language models. Typically there are different dimensions for evaluation.
|       Metrics       | Better indicator |
| :-----------------: | :--------------: |
|      Fluency       |    $\uparrow$    |
|    Relevance    |    $\uparrow$    |
|       Knowledge        |    $\uparrow$    |
|        Correctness        |    $\uparrow$    |

## Related Project
TBD
