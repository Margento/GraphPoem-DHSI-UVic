# GraphPoem-DHSI-UVic


## OVERVIEW, GOALS, & OUTPUTS
This research looks creatively, retrospectively, and speculatively at the documented history of #GraphPoem @ DHSI @ UVic (2019–2024). Its records comprise evolving corpora of text and/as media, participant data-commoning logs across Facebook, Twitter, and—most consequentially—JupyterHub, Python scripts, and webformance livestreams uploaded to YouTube.


Three principal strands emerged from these records, corresponding, perhaps, to the three most defining dimensions of #GraphPoem @ DHSI: the institutional, the communal, and the personal. More specifically, these strands concern DHSI as the institute proper; the #GraphPoem participants and viewers; and the poet(ry collective) MARGENTO, respectively.


Using an in-house text-to-audio-to-video feature-mapping system (https://github.com/Margento/graph-poem) the project examined the coherence between the records of these three strands and their respective “futures.” For now, these futures consist of text data from DHSI @ UdeM (2025–present) and, at the personal level, in-progress sequences of poems inspired by #GraphPoem’s transition from Victoria, BC, to Montreal, QC.
Both records and futures were mapped onto clips from the livestreams they displayed the greatest coherence with. This mapping enabled the project’s computationally speculative dimension.


The resulting analysis suggested that institutional records evolve intermedially toward personal futures, while personal corpora gradually become submerged within institutional future trends. Throughout this process, the communal remains on course, acting as a dynamic bridge between the two.


The resulting videopoem samples records from each strand, interspersing them with flashforwards that emerge according to probabilistic distributions that elude extractivist learners. The project thus stages a speculative encounter between computational patterning, intermedial memory, and possible futures.


## METHODOLOGY
The retrospective textual components of the three dimensions were mapped to most similar clips in the recordings based on affect vectors around scene changes or cuts. Then similarities between the latter and the textual futures were computed based not on affect, but on basic (poetic and environmental/musical) acoustic and temporal features (https://github.com/Margento/graph-poem). This was an opportunity to probe whether the three strands stayed the course or intermingled in their projected evolution (by checking whether the textual futures evolved towards the same clips as their retrospective counterparts in the wider environment of all livestream recordings). The outputs told us which clips and texts (from which future) will jam which strand’s retrospective record.


The last task was then to figure out exactly where in the audio-video of each strand will the clips 'from the future' be inserted. I called an open-source LLM (llama 3.2) to make suggestions in that respect while taking into account two related subtasks: first, the arrangement---through the pattern emerging from it---needs to resonate with the specific ethos of each strand within the wider framework of #GraphPoem as a whole, and, second, the distribution of these insertions has to be unlearnable to  (a long-standing anti-extractivist #GraphPoem mainstay, see https://www.routledge.com/Literature-and-Computation-Platform-Intermediality-Hermeneutic-Modeling-and-Analytical-Creative-Approaches/Tanasescu/p/book/9781032341675 & https://github.com/Margento/GraphPoem-DHSI). 



## TECHNICAL PIPELINE

The project proceeds through a sequence of transformations that moves from textual and audiovisual archives to feature representations, similarity mappings, clip selection, speculative insertions, and, finally, videopoetic assembly. The pipeline is designed so that computational outputs remain available for interpretation and artistic intervention rather than functioning as an entirely automated editing system.


At a high level, the workflow can be represented as:


```text
ARCHIVAL MATERIAL
        ↓
TEXTUAL FEATURE EXTRACTION
        +
LIVESTREAM / SHOT FEATURE EXTRACTION
        ↓
RETROSPECTIVE TEXT → VIDEO MAPPING
        ↓
FUTURE TEXT → CANDIDATE CLIP MAPPING
        ↓
STRAND / FUTURE COHERENCE ANALYSIS
        ↓
CLIP EXTRACTION
        ↓
LLM-SUGGESTED INSERTION ORDER & DISTRIBUTION
        ↓
AUDIO-VIDEO INSERTION & VERIFICATION
        ↓
USER / ARTISTIC REVIEW
        ↓
FINAL VIDEOPOEM
```


### 1. Prepare the corpora

The pipeline begins with the project's archival materials, which are organized around three principal strands:


* **the institutional**, concerning DHSI;

  
* **the communal**, concerning #GraphPoem participants, viewers, and their data-commoning practices;

  
* **the personal**, concerning the poet(ry collective) MARGENTO.

These retrospective records are placed in relation to corresponding textual “futures.” At present, these include materials associated with DHSI @ UdeM and ongoing poetic sequences emerging from #GraphPoem's transition from Victoria to Montreal (nomadosophically via Brussels, Hong Kong, Barcelona, and Galway).


The textual materials are processed as corpora rather than treated only as semantic documents. Their formal, sonic, affective, and temporal properties become computational features that can subsequently be compared with features extracted from audiovisual material.


### 2. Extract textual, affective, acoustic, and temporal features


For each poem or textual unit, the pipeline derives several feature layers.


The textual analysis includes approximate syllable counts, syllable density, pacing and pacing variation, phonological features, recurring motifs, enjambment and caesura measures, and the identification of structural ruptures or unusually divergent lines. These features come with a bijective mapping to audio & video features.


Affect is represented through a three-dimensional vector:


```text
[valence, arousal, energy]
```


Valence is derived from multilingual sentiment analysis, while arousal and energy are calculated in relation to audio-like textual features, including pacing variance, fricative density, silence ratio, tempo, and syllable density.


Temporal analysis treats lines as segments and records such information as:


* the number of segments;
* words and syllables per segment;
* repeated phonological clusters;
* recurrent motifs;
* statistically unusual rhythmic or syllabic deviations;
* linear, cyclical, recursive, and hybrid structural tendencies.
[all of the above are also mapped to audio/video features]


The result is a structured representation of each textual object that can be compared across corpora and across media.


### 3. Identify candidate moments in the livestream archive


The audiovisual archive consists of #GraphPoem webformance livestream recordings.


Rather than treating every frame as equally significant, the pipeline concentrates on candidate moments around **scene changes or cuts**. Individual shots or temporally localized windows are represented through extracted audiovisual features and stored together with identifying information such as source year, shot number, and timestamp.


These candidate moments form the searchable audiovisual environment into which the retrospective and future textual materials are mapped.


Where practical, the workflow can operate on lower-resolution or sparsely sampled proxies during feature extraction, preserving the original recordings for final clip production. Feature tables, indexes, similarity results, and extracted shots then become the primary working representations, reducing the need to repeatedly process the original videos.


### 4. Map retrospective records onto audiovisual material through affect


The retrospective textual components of the institutional, communal, and personal strands are first mapped onto the livestream archive through affective similarity.


For each textual unit, its affect vector is compared with the affective characteristics associated with candidate audiovisual moments, especially around scene changes and cuts.


Conceptually:


```text
retrospective text
        ↓
[valence, arousal, energy]
        ↓
similarity comparison
        ↓
ranked livestream shots / clips
```


The highest-ranking results identify clips that are most coherent, according to the project's feature model, with the retrospective record under consideration.


This stage establishes the audiovisual reference points from which each strand's subsequent trajectory can be examined.


### 5. Map the textual futures through acoustic and temporal features


The second mapping deliberately uses a different feature space.


Instead of relying primarily on affect, the textual futures are compared with the candidate audiovisual environment through basic **poetic and environmental/musical acoustic features** and **temporal structures**.


This makes it possible to ask whether the future material gravitates toward the same audiovisual regions as the retrospective material or whether its trajectory moves elsewhere.


In simplified form:


```text
retrospective text
        ↓ affect similarity
        ↓
candidate clip(s)
        ↑
acoustic + temporal similarity
        ↑
future text
```


The resulting comparisons indicate which clips are most relevant to both a strand's retrospective record and its projected future, and therefore which textual materials and audiovisual moments can be brought into relation—-or, in the project's terms, **jammed** together.


### 6. Compare trajectories across the three strands


The mappings are then interpreted comparatively.


The project examines whether the institutional, communal, and personal trajectories remain distinct, converge, interfere with one another, or move toward unexpected audiovisual regions.


The communal strand is particularly important here as a possible dynamic bridge between institutional and personal trajectories.


The outputs of this stage therefore do not only answer the question: Which clip is most similar to this text?


They also (and mainly) aim to support broader questions such as:


* Do retrospective and future materials converge on similar clips?

  
* Does one strand evolve toward another?

  
* Does a future belonging to one strand become more coherent with the audiovisual history of another?

  
* Which relationships remain stable, and which become unstable?


These comparisons provide the computational basis for the project's speculative dimension.


### 7. Rank and extract the selected clips


Similarity results are stored in structured output files, including ranked JSON results containing:

* source folder and year;

  
* image or shot path;

  
* shot name;

  
* timestamp;

  
* similarity score;

  
* associated feature values.


The clip-extraction stage reads the highest-ranking entries, parses the source year and timestamp from their paths, and uses **FFmpeg** to recover the corresponding segments from the original livestream recordings.


Clips are encoded with video and audio streams and written as standalone media files for subsequent assembly.


A typical extraction process is therefore:


```text
ranked similarity JSON
        ↓
source year
        +
timestamp
        ↓
locate original livestream
        ↓
FFmpeg clip extraction
        ↓
selected audiovisual segment
```


This stage reconnects the abstract feature-space result with the original high-resolution audiovisual archive.


### 8. Determine where the “future” enters the retrospective record


Once the relevant clips have been identified, the remaining problem is not only simply which clips to use, but **where they should enter the existing audio-video sequence**.


For this task, the repository uses a locally running **Llama 3.2** model through Ollama.


The LLM receives similarity information describing the relationship between candidate clips and existing shots and is asked to propose:


* an ordering of selected clips;

  
* a rationale for that ordering;

  
* a probability distribution governing their insertion.


The resulting JSON files preserve the model's suggested sequence and probabilities so that these recommendations remain inspectable rather than disappearing into the editing process.


The LLM is therefore not used to determine the project's conceptual argument or to replace artistic judgment. Its role is more specific: to generate proposals for an arrangement whose emerging pattern can resonate with the ethos of a particular strand while avoiding a straightforwardly predictable distribution.


### 9. Introduce probabilistic and anti-extractivist distributions


The placement of clips “from the future” is designed not to follow a simple, easily learnable periodic rule.


Instead, insertion positions are distributed according to probabilities associated with the selected sequence. This produces flashforwards whose appearance can remain uneven, contingent, and resistant to simple extraction as a reusable pattern.


### 10. Insert the clips into the source video


The selected future clips are then inserted into the appropriate audiovisual positions.


The insertion scripts construct the final video through FFmpeg-based processing while preserving the relationship between the source video's duration and its audio stream.


The final output is checked for:


* successful processing;

  
* output duration;

  
* correspondence with the original duration where appropriate;

  
* presence of an audio stream;

  
* FFmpeg errors or failures.
  

This verification stage ensures that computationally generated edit decisions produce a technically usable audiovisual output.


### 11. Return to user viewing and artistic assembly

The computational pipeline ends by returning the results to viewing, listening, interpretation, and editing.


Similarity scores and LLM-generated orderings are proposals, not final aesthetic decisions. The selected clips can be reviewed, reordered, rejected, replaced, or further manipulated during the assembly of the videopoem.


The full workflow is therefore not (only) about automated video generation but aims to involve a cycle of computational suggestion and analytical-creative intervention:


```text
archive
    ↓
feature extraction
    ↓
similarity mapping
    ↓
candidate retrieval
    ↓
computational suggestions
    ↓
user viewing / listening
    ↓
selection and revision
    ↓
montage
```


The interpretive encounter with the proposed material is an integral part of the methodology.


---


## HOW TO USE THIS PIPELINE


The repository contains project-specific scripts and data products rather than a single universal command-line application. As such, reproducing the workflow involves preparing compatible inputs and running the relevant stages in sequence.


### Requirements

The pipeline uses Python together with the libraries required by the individual scripts. Depending on the stage being run, these include packages for:

* numerical processing;

  
* JSON and filesystem handling;

  
* computer vision and image processing;

  
* text and feature extraction;

  
* HTTP requests;

  
* multimedia/multimodal processing.


The audiovisual stages require **FFmpeg**.


The clip-ordering stage additionally requires:


* Ollama;

  
* a locally available `llama3.2` model;

  
* a running Ollama server;

  
* the Python `requests` package.
  

The repository's Llama integration expects the Ollama API to be available locally.


### Step 1: Organize the source material


Place the relevant textual corpora and livestream recordings in the expected working directories.


Maintain the relationship between:


```text
textual corpus
        ↔
strand
        ↔
retrospective or future status
```


Likewise, preserve recognizable naming conventions for the extracted audiovisual material, since later scripts recover information such as DHSI year and timestamps from file paths and shot names.


### Step 2: Run or reuse textual feature extraction


Process the retrospective and future texts to generate feature representations.


The output should preserve, at minimum, the feature layers required by the comparison stages:


```text
audio / poetic features
affect vector
temporal features
motifs
structural ruptures
```


These representations become the inputs to the similarity calculations.


### Step 3: Prepare the livestream archive


Extract or generate candidate shots and temporal windows from the livestream recordings.


Each candidate should retain enough metadata to recover its source:


```text
source folder / year
shot identifier
timestamp
feature representation
```


The repository's existing naming conventions allow scripts to infer the DHSI year and the time position of a selected shot.


### Step 4: Calculate retrospective text-to-video similarities


Compare the affective representations of the retrospective textual materials with the candidate audiovisual moments.


Store the ranked results in a structured format such as JSON so that each result records both its similarity score and the information needed to retrieve the corresponding clip.


### Step 5: Calculate future-oriented similarities


For the textual futures, perform the acoustic and temporal comparison described above.


Use the resulting scores to identify candidate clips that reveal whether future materials remain coherent with, diverge from, or intermingle with the audiovisual trajectories associated with the retrospective strands.


### Step 6: Extract the highest-ranking clips


Run the clip-extraction script on the ranked similarity results.


The script reads the selected entries, identifies their source videos and timestamps, and invokes FFmpeg to create standalone clips with audio.


Adjust the number of top-ranked entries or the clip duration according to the scale of the montage being assembled.


### Step 7: Run the Llama 3.2 ordering stage


Install Ollama, make sure the Llama 3.2 model is available, and start the local Ollama service.


Then run the appropriate clip-ordering script.


The script:


1. loads a similarity-result JSON file;

  
3. formats the relevant candidate and shot information into a prompt;

   
5. sends the request to the local Ollama API;

   
7. requests an ordering, rationale, and probability distribution;

   
9. saves the resulting response as JSON.
   

The saved JSON can then be inspected and, where necessary, manually edited before proceeding.


### Step 8: Insert and assemble the clips


Use the insertion and FFmpeg-based assembly scripts to place the selected clips into the relevant audiovisual sequence.


The insertion positions should be informed by the ordering and probability structure generated in the preceding stage, while remaining open to artistic revision.


### Step 9: Verify the output


After rendering, check:


* whether the process completed successfully;

  
* the final video's duration;

  
* whether the expected audio stream is present;

  
* whether the inserted clips appear at the intended locations.
  

If necessary, revise the ordering, probabilities, selected clips, or insertion logic and render again.


### A recommended way to begin


For experimentation or adaptation, it is advisable to begin with a deliberately small subset of the archive rather than processing all years, all texts, and all platforms at once.


A manageable initial experiment might consist of:


```text
one DHSI year
×
one livestream
×
a limited number of significant textual regions
```


The resulting candidates can then be watched and evaluated before scaling the workflow to the larger archive.


---


## ACKNOWLEDGEMENTS & CREDITS


This project emerges from and remains indebted to the many people, performances, platforms, archives, and collaborative practices that have constituted **#GraphPoem @ DHSI**.

### #GraphPoem and its participants

Our deepest thanks go to the participants, viewers, collaborators, coders, performers, and data-commoners whose interactions across the project's various iterations made this archive possible.

The communal dimension of the project is not simply a dataset or an object of analysis. It is one of the conditions of possibility of the work itself.

### DHSI

We gratefully acknowledge the **Digital Humanities Summer Institute (DHSI)** and the institutional contexts in which #GraphPoem developed, including its history at the University of Victoria and its continuing transformations.


The project's retrospective and speculative trajectories are inseparable from these institutional environments and from the communities that formed within and around them.


Our deepest thanks and gratitude go particularly to the Founding Director of DHSI, Prof. Ray Siemens, who full-heartedly supported and encouraged this initiative from its very inception and throughout the UVic years. A huge thank you to our colleagues at the Electronic Textual Cultures Lab (ETCL) and the members of all annual DHSI teams. 


### MARGENTO


The personal strand of the project draws on the ongoing poetic, performative, and computational work of the artist/poet/coder (collective) **MARGENTO**.


Music: MARGENTO---Costin Dumitrache & Valentin Baicu; vocals: Marina Gingiroff, Maria Raducanu, Sorina Enea ; Abis---Costin Dumitrache; vocals: Marina Gingiroff


Action painting: Grigore Negrescu (MARGENTO)


Coding: Chris Tanasescu (MARGENTO)


JupyterHub Administration & Coding support: Prasadith Buddhitha Kirinde Gamaarachchige (uOttawa; MARGENTO)


NLP expert consultant: Prof. Diana Inkpen (uOttawa)


Poems, excerpts, (&/or lyrics) featured in the videpoem by Elke de Rijcke, Carl Norac, David Baker, Sappho, Ion Barbu, Chris Tanasescu / MARGENTO, & Marina Gingiroff (Abis).

The project also builds upon the text-to-audio-to-video feature-mapping methods developed through the broader GraphPoem work, particularly https://github.com/Margento/graph-poem (and in the "futures" part, https://github.com/Margento/Sympoiesis).



### Open-source software and computational tools

The technical workflow makes use of open-source software and libraries, including:

* Python and its scientific and data-processing ecosystem;

  
* OpenCV and related computer-vision tools;

  
* FFmpeg for audiovisual processing and clip extraction;

  
* Hugging Face tooling and multilingual sentiment analysis;

  
* Ollama for local model execution;

  
* Meta's Llama 3.2 model, used here for locally generated suggestions concerning clip ordering and probabilistic distribution.


These tools are used as components within an analytical-creative process rather than as autonomous authors of the resulting videopoem.


### Related projects and research


This repository builds on the broader methodological and conceptual history of #GraphPoem and on related work concerning literature, computation, platform intermediality, hermeneutic modeling, and analytical-creative approaches.


For further context, see the related GraphPoem repositories and the research cited in the project's existing overview and methodology.

---

## NOTES ON REUSE AND ADAPTATION

This pipeline is both project-specific and adaptable.

Some elements—particularly directory names, file naming conventions, corpus definitions, similarity inputs, and insertion targets—are tied to the present #GraphPoem @ DHSI archive and will need to be modified when working with another corpus. To run the scripts, replace the name of the folder int he current setup, replace the name of the folder “costin_graphpoem_dhsi_shots” with “personal_graphpoem_dhsi_shots” 
& “after_that_graphpoem_dhsi_extracted_clips” with “after_that_community_graphpoem_dhsi_extracted_clips” 


The broader sequence, however, can be adapted:


```text
text / archival material
        ↓
multimodal feature extraction
        ↓
cross-media similarity mapping
        ↓
candidate retrieval
        ↓
probabilistic ordering
        ↓
human interpretation
        ↓
audiovisual montage
```


The central principle is to keep the intermediate computational representations visible and revisable. Feature vectors, similarity rankings, selected clips, LLM suggestions, and probability distributions should remain inspectable parts of the research process rather than becoming invisible operations behind a final media object.


⚖️ License Note: The code in this repository is open-source under the MIT License. The text, data, and media assets are copyrighted by the repository owner and may only be downloaded and shared in their original form with proper attribution. See the LICENSE file for details.
