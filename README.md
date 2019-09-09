# CPSC 532P / LING 530A: Deep Learning for Natural Language Processing (DL-NLP)
### The University of British Columbia

**Year:** Winter Session I 2019

**Time:** Tue Thu 14:00 15:30.

**Location:** The Leon and Thea Koerner University Centre (UCLL) (Right by Rose Garden at 6331 Crescent Road V6T 1Z1). Room 109.

**Instructor:** Dr. Muhammad Abdul-Mageed

**Office location:** Totem Field Studios 224 (Department of Linguistics: 2613 West Mall V6T 1Z4)

**Office phone:** (Apologies, I do not use office phone. Please email me)

**Office hours:** Tue. 12:00-14:00pm @Totem Field Studios 224, or by appointment. *(I can also handle inquiries via email or, in limited cases, Skype.)*


**E-mail address:** muhammad.mageed@ubc.ca

**Student Portal:** <http://canvas.ubc.ca>


## 1.	Course Rationale & Goal: 

**Rationale/Background:** *Deep learning* is a class of machine learning methods
inspired by information processing in the human brain, whereas *Natural language
processing (NLP)* is the field focused at teaching computers to understand and generate
human language. Emotion detection where a program is able to identify the
type of expressed emotion from language is an example of language understanding.
Dialog systems where the computer interacts with humans, such as the Amazon
Echo, constitute an instance of both language understanding and generation, as
the machine identifies the meaning of questions and generates meaningful answers.
Other examples of NLP include speech processing and machine translation. ***Deep
learning of natural language is transformative***, and has recently broken
records on several NLP tasks. The field is also in its infancy, with ***fascinating
future breakthroughs ahead***. Solving NLP problems directly contributes
to the development of pervasive technologies with significant social and economic
impacts and the potential to enhance the lives of millions of people. ***Given the
central role that language plays in our lives, this research has implications
across almost all fields of science and technology***, as well as other
disciplines, as NLP and deep learning are instrumental for making sense of the
ever-growing data collected in these fields.

**Goal:** TThis course provides a graduate-level introduction to deep learning,
with a focus on NLP problems and applications. The goal of the course is to
familiarize students with the major deep learning methods and practices. This
includes, for example, how neural networks are trained, the core neural network
architectures, and the primary deep learning methods being developed to solve
language problems. This includes problems at various linguistic levels (e.g., word
and sub-word, phrase, clause, and discourse). For example, we will cover unsupervised,
distributed representations and supervised deep learning methods across
these different linguistic levels. Through homework and a final project, the course
also provides a context for hands-on experience in using deep learning software to
develop advanced solutions for NLP problems.

**Potential audiences for this course are:**

* People with a linguistics, computer science, and/or engineering background
interested in learning novel deep learning and NLP methods.

* People with other machine learning backgrounds interested in deep learning
and/or NLP.

## 2.	Course Objectives: 
Upon completion of this course students will be able to:


* *identify* the core principles of training and designing artificial neural networks
* *identify* the inherent ambiguity in natural language, and appreciate challenges
associated with teaching machines to understand and generate it
* *become* aware of the major deep learning methods being developed for solving
NLP problems, and be in a position to apply this deepened understanding
in critical, creative, and novel ways
* *become* aware of a core of NLP problems, and demonstrate how these are
relevant to the lives of diverse individuals, communities and organizations.
* *collaborate* effectively with peers through course assignments
* *identify* an NLP problem (existing or novel) and apply deep learning methods
to develop a novel solution for it

## 3.	Course Topics:

* word, phrase, and sentence meaning
* feedforward networks
* recurrent neural networks 
* convolutional neural networks
* language models
* seq2seq models
* attention & Transformers
* deep generative models (auo-encoders & generative adversarial networks)

### Applications ###

* machine translation
* controlled language generation
* summarization
* image and video captioning
* morphosyntax (e.g., POS tagging and morphological disambiguation)
* text classification (e.g., sentiment analysis, emotion detection, language)

## 4.	Prerequisites:  

* familiarity with basic linear algebra, basic calculus, and basic probability
(basic = high school level)
* have programming experience in Python
* familiarity with at least one area of linguistics
* have access to a computer with a GPU on a regular basis
* ability to work individually as well as with team
 
***Students lacking any of the above pre-requisites must be open to
learn outside their comfort zone to make up, including investing time
outside class learning these pre-requisites on their own. Some relevant
material across these pre-requisites will be linked from the syllabus.
Although very light support might be provided, the engineering is exclusively
the students’ responsibility. This course has no lab section.
Should you have questions about pre-requisites, please email the instructor.***
 
## 5.	Format of the course: 
•	This course will involve lectures, class hands-on activities, individual and group work, and instructor-, peer-, and self-assessment. 

## 6. Course syllabus:

**Recommended books:**

* Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). Deep learning (Vol. 1). Cambridge: MIT press. Available at: [[link](http://www.deeplearningbook.org/)]. 

* Jurafsky, D., & Martin, J. H. (2017). Speech and language processing. London:: Pearson. Available at [[link](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf)].

**Other related material:**

* Bird, S., Klein, E., & Loper, E. (2009). Natural language processing with Python. O'Reilly Media, Inc. [[link](http://www.nltk.org/book/)].
* Weekly readings will be assigned from materials available through the UBC library and online.
* See "Readings Section below"

## 7. Calendar / Weekly schedule (tentative)

| Date | Slides   | Related Content                 | What is due/out            |
|------|--------|----------------------------------|----------------------------|
| Mon Sept 3    |  No class; TA Training  | NA |            |
| Wed Sept 5    |  Course overview  | [overview_slides](https://github.com/UBC-NLP/dlnlp2019/blob/master/slides/intro_deeplearning.pdf)   |            |
| Mon Sept 10    |  Probability | [prob_slides](https://github.com/UBC-NLP/dlnlp2019/blob/master/slides/probability.pdf); [[DLB CH03](https://www.deeplearningbook.org/contents/prob.html); [KA](https://www.khanacademy.org/math/statistics-probability/probability/probability-geometry); [Harvard Stats](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo)]  |            |
| Wed Sept 12    |  Math Refershers II |  |   hw01 out (Canvas)         |
| Mon Sept 17    |    Information Theory  |        |
| Wed Sept 19    |    ML Basics |         |
| Mon Sept 24    |  Word meaning |  |         |
| Wed Sept 26    |    word vectors & Word2Vec |  | | 
| Mon Oct 1    |     Language Models I |  |  hw01 due & hw02 out |
| Wed Oct 3    |    Feedforward Networks | | |
| Mon Oct 8    |     Recurrent Neural Networks |   |            |
| Wed Oct 10    |     LSTMs |    |          |
| Wed Oct 17    |   Transformers |  |          |
| Mon Oct 22   |     Applications & Project Discussion |  | |        |
| Wed Oct 24   |     Gradient-based optimization |   |         |
| Mon Oct 29    |    Applications I |  |        |
| Wed Oct 31    |    Seq2Seq; Neural Machine Translation |   |   hw02 due     |
| Mon Nov 5    |     ConvNets I   |   |   hw03_a due        |
| Wed Nov 7    |    ConvNets II | |      |
| Mon Nov 12    |    |   |            |
| Wed Nov 14    |    |  |         |
| Mon Nov 19    | BERT  |      |          |
| Mon Nov 26    |    TansformerXL & XLNet    |        | |
| Wed Nov 28    |    Projects  |        |        |
| Mon Dec 3    |    | --  |   Final project due  (hw03_b)       |



## 8. Readings:  
See Section 7 above.
Additionally, below is list of relevant/interesting/background papers.
This list will grow soon.

* [[Toward controlled generation of text](http://www.cs.ubc.ca/~amuham01/LING530/papers/hu2017toward.pdf)]

* [[Tutorial on variational autoencoders](http://www.cs.ubc.ca/~amuham01/LING530/papers/doersch2016tutorial.pdf)]

* [[Adversarially Regularized Autoencoders](http://www.cs.ubc.ca/~amuham01/LING530/papers/zhao2018adversarially.pdf)]

* [[Joint Embedding of Words and Labels for Text Classification](http://www.cs.ubc.ca/~amuham01/LING530/papers/wang2018joint.pdf)]

* [[Learning Adversarial Networks for Semi-Supervised Text Classification via Policy Gradient](http://www.cs.ubc.ca/~amuham01/LING530/papers/li2018learning.pdf)]

* [[Realistic Evaluation of Semi-Supervised Learning Algorithms](http://www.cs.ubc.ca/~amuham01/LING530/papers/oliver2018realistic.pdf)]

* [[Improving language understanding by generative pre-training](http://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)]

* [[Automatic Stance Detection Using End-to-End Memory Networks](http://www.cs.ubc.ca/~amuham01/LING530/papers/mohtarami2018automatic.pdf)]

* [[Neural Machine Translation by Jointly Learning to Align and Translate](http://www.cs.ubc.ca/~amuham01/LING530/papers/bahdanau2014neural.pdf)]

* [[Grammar as a foreign language](http://www.cs.ubc.ca/~amuham01/LING530/papers/vinyals2015grammar.pdf)] 

* [[Deep Models for Arabic Dialect Identification on Benchmarked Data](http://www.cs.ubc.ca/~amuham01/LING530/papers/elarabyDeepModels2018.pdf)]

* [[A Neural Model for User Geolocation and Lexical Dialectology](http://www.cs.ubc.ca/~amuham01/LING530/papers/rahimiGeoloc2017.pdf)]

* [[Deep Contextualized Word Representations]](http://www.cs.ubc.ca/~amuham01/LING530/papers/petersELMo2018.pdf)

* [[Emonet: Fine-grained emotion detection with gated recurrent neural networks]](http://www.cs.ubc.ca/~amuham01/LING530/papers/mageedEmoNet2017.pdf)

* [[Is statistical machine translation approach dead?](http://www.cs.ubc.ca/~amuham01/LING530/papers/menacer2017statistical.pdf)]

* [[Neural machine translation and sequence-to-sequence models: A tutorial](http://www.cs.ubc.ca/~amuham01/LING530/papers/neubig2017neural.pdf)]

* [[Adversarial training methods for semi-supervised text classification](http://www.cs.ubc.ca/~amuham01/LING530/papers/miyato2016adversarial.pdf)]

* [[Learned in translation: Contextualized word vectors](http://www.cs.ubc.ca/~amuham01/LING530/papers/mccann2017learned.pdf)]

* [[Semi-supervised sequence learning](http://www.cs.ubc.ca/~amuham01/LING530/papers/dai2015semi.pdf)]

* [[Auto-encoding variational bayes](http://www.cs.ubc.ca/~amuham01/LING530/papers/kingma2013auto.pdf)]

* [[Sentiment Transfer using Seq2Seq Adversarial Autoencoders](http://www.cs.ubc.ca/~amuham01/LING530/papers/singh2018sentiment.pdf)]

* [[Variational Autoencoder for Semi-Supervised Text Classification](http://www.cs.ubc.ca/~amuham01/LING530/papers/xu2017variational.pdf)]

* [[Phrase-Based \& Neural Unsupervised Machine Translation](http://www.cs.ubc.ca/~amuham01/LING530/papers/lample2018phrase.pdf)]

* [[Asma: A system for automatic segmentation and morpho-syntactic disambiguation of modern standard arabic](http://www.aclweb.org/anthology/R13-1001)]





## 9. Course Assignments/Grades:
| Assignment | Due date | Weight |
| ---------- | -------- | ------ |
| Professionalization & Class Participation |  Throughout | 15% |
| ASSIGNMENT 1: Individual assignment: Word Embeddings |  Oct 3 | 15% |
| ASSIGNMENT 2: Individual assignment: Text Classification with RNNs |  Oct. 24 | 20% |
| ASSIGNMENT 3_A: Group assignment: Project Proposal |    Nov. 5 | 5% |
| ASSIGNMENT 3_B: Group assignment: Term Project (GROUP of 3) |   Dec 3   | 45% |

## Notes on Assignments:


### ASSIGNMENT 1: *Word Embeddings* 

In this assignment, students will train a word embeddings model (using either Word2Vec of FastText) on a dataset provided to them, and use the trained model for a number of tasks. The students will write a report using the Jupyter platform describing their work. The reprot will include both the code used and the results acquired, in a step-wise, organized fashion. The students are expected to use enabling Python libraries (e.g., NLTK, gensim, scikit-learn) and illustrate their work with visualizations automatically generated from the data. A rubric will be provided to walk students through the different steps. 

**Deliverables: *Jupyter Notebook describing the work students carried out***

### ASSIGNMENT 2: *Text Classification with RNNs* 

For this assignment, students will build a fully-fledged text classifier using RNNs, and its variations (e.g., LSTMs, GRUs). For this, the students will be provided a labeled dataset. As such, the task is cast as supervised machine learning. The students will need to demonstrate understanding of the theoritical part of the course. For example, a detailed description of pre-processing, data splits (concepts like train, dev, and test sets and/or n-fold cross validation), model capacity (e.g., the architecture of the network used in terms of layers and number of units in each layer), activation functions, regularization, cost functions, baselines and meaningful comparisons (which baselines are chosen and why), etc. 

**Deliverables: *A short paper (4 pages main body+ 2 pages references) + Jupyter notebook***

Students will then write a 4-page report + 2 extra pages for references. A maximum of 2 extra pages can be allowed, without instructor permission, but need to be justfied in a footnote in the report. Students will also need to submit an accompanying Jupyter notebook with their code. Students will be provided an ACL LaTex template to use for writing the report, and a rubric walking them through some of the major details of how to write the report.

### ASSIGNMENT 3: *Term Project*

The purposes of this assignment include:
- Identifying and describing an interesting (i.e., challenging, novel, socially-relevant, has practical applications) NLP problem;
- Reviewing previous literature on relevant to the task, showing a clear understanding of previous scholarship;
- Proposing and developing (in the engineering sense via code) a solution to the problem using deep learning;
- Clearly describing the work, justifying decisons made, in a way that lends work to replication;
- Developing oral and written communication skills through discussions with classmates and instructor;
- Demonstrating ability to work as part of a team, including initiative taking, integrity, dependability and co-operation, and effective collaboration.

For this assignment, each student is required to work as part of a group of of 3\* on a project involving a practical task. Example projects will be discussed in class.

* A group of a different size may be possible after consultation with the instructor.

**Deliverables** 
**Proposal: 1-2 pages in ACL Style LaTex**
- Who are the the group members?
- What is the problem you are proposing a solution for?
- What motivates your work? Why is it important or useful to undertake the chosen task?
- How does your project compare to NLP and deep learning published research?
- What are the different steps you will take to ensure success of the project? What are the smaller segments of which the bigger task is composed? And how will you conduct each small task?
- How does the work bread down and what each member of the team be contributing?
- Timeline for completing the project, including goals for each segment. 
Note: You may like to provide a table breaking down who is doing what and when.
The LaTex files will be provided to you as part of the rubric for the final project.

**Final Paper: 6-8 pages (ACL Style)** 
The final deliverable includes:
- A detailed and clear description of your project, including the necessary sections, as appropriate. For example, you will need to include an abstract, introduction, possibly research questions, a literature review, a description of dataset (collection, splits), methods (implementation details, experimental conditions, settings, comparisons, baselines), results (in tables), analysis of results (e.g., with visulaizations, error analyses), and a conclusion describing limitations and future directions;
- All data used, whenever possible, and/or links to where the data are housed;
- Pointers to a live version of the project, if any;
- As appropriate, you should situate your work within the wider context of published works and approaches, with supporting arguments. You will have an unlimited number of pages for references, but you should plan to have at least 15 sources;
- Employment of figures, tables, and visualizations as appropriate to enhance argument and facilitate communicating your findings/results;

## 10. Course Policies

**Attendance:** The UBC calendar states: “Regular attendance is expected of students in all their classes (including lectures, laboratories, tutorials, seminars, etc.). Students who neglect their academic work and assignments may be excluded from the final examinations. Students who are unavoidably absent because of illness or disability should report to their instructors on return to classes.”

**Evaluation:** All assignments will be marked using the evaluative criteria given in this syllabus.

**Access & Diversity:** Access & Diversity works with the University to create an inclusive living and learning environment in which all students can thrive. The University accommodates students with disabilities who have registered with the [Access and Diversity unit](https://students.ubc.ca/about-student-services/access-diversity). You must register with the Disability Resource Centre to be granted special accommodations for any on-going conditions. 
**Religious Accommodation:** The University accommodates students whose religious obligations conflict with attendance, submitting assignments, or completing scheduled tests and examinations. Please let your instructor know in advance, preferably in the first week of class, if you will require any accommodation on these grounds. Students who plan to be absent for varsity athletics, family obligations, or other similar commitments, cannot assume they will be accommodated, and should discuss their commitments with the instructor before the course drop date. [UBC policy on Religious Holidays](http://www.universitycounsel.ubc.ca/policies/policy65.pdf).

## Academic Integrity
**Plagiarism**
Plagiarism is the most serious academic offence that a student can commit. Regardless of whether or not it was committed intentionally, plagiarism has serious academic consequences and can result in expulsion from the university. Plagiarism involves the improper use of somebody else's words or ideas in one's work. 

It is your responsibility to make sure you fully understand what plagiarism is. Many students who think they understand plagiarism do in fact commit what UBC calls "reckless plagiarism." Below is an excerpt on reckless plagiarism from UBC Faculty of Arts' leaflet, [Plagiarism Avoided: Taking Responsibility for Your Work](http://www.arts.ubc.ca/arts-students/plagiarism-avoided.html).

"The bulk of plagiarism falls into this category. Reckless plagiarism is often the result of careless research, poor time management, and a lack of confidence in your own ability to think critically. Examples of reckless plagiarism include:

* Taking phrases, sentences, paragraphs, or statistical findings from a variety of sources and piecing them together into an essay (piecemeal plagiarism);
* Taking the words of another author and failing to note clearly that they are not your own. In other words, you have not put a direct quotation within quotation marks;
* Using statistical findings without acknowledging your source;
* Taking another author's idea, without your own critical analysis, and failing to acknowledge that this idea is not yours;
* Paraphrasing (i.e. rewording or rearranging words so that your work resembles, but does not copy, the original) without acknowledging your source;
* Using footnotes or material quoted in other sources as if they were the results of your own research; and
* Submitting a piece of work with inaccurate text references, sloppy footnotes, or incomplete source (bibliographic) information."

Bear in mind that this is only one example of the different forms of plagiarism. Before preparing for their written assignments, students are strongly encouraged to familiarize themselves with the following source on plagiarism: the [Academic Integrity Resource Centre](http://help.library.ubc.ca/researching/academic-integrity).

If after reading these materials you still are unsure about how to properly use sources in your work, please ask me for clarification. Students are held responsible for knowing and following all University regulations regarding academic dishonesty. If a student does not know how to properly cite a source or what constitutes proper use of a source it is the student's personal responsibility to obtain the needed information and to apply it within University guidelines and policies. If evidence of academic dishonesty is found in a course assignment, previously submitted work in this course may be reviewed for possible academic dishonesty and grades modified as appropriate. UBC policy requires that all suspected cases of academic dishonesty must be forwarded to the Dean for possible action. 

