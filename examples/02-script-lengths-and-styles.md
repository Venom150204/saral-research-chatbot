# Script lengths and styles

Source: `eval/datasets/decks/attention.pptx`. Recorded by `python -m eval.run_eval --only examples` on 2026-10-14; nothing edited by hand.

---

**User:** Make a 90-second script for policymakers.

*Tools: write_deliverable, write_deliverable*

**SARAL:** I am writing a 90-second script for policymakers.

I have created the script for policymakers, though it runs about 75 seconds, short of the 90 you asked for `[slide 2, slide 2]`. Both requirements—tailoring it for policymakers and including speaker cues—were met. Would you like me to expand it further to hit the full 90-second duration?

*Draft version 1 · status: succeeded*

**#1 Ninety-Second Policy Briefing on Transformer AI Efficiency**

- *speaker_cue*: [Presenter steps to the podium, addressing policy leaders and gesturing to the main display showing global computing trends.] `(framing)`
- *spoken_text*: For years, artificial intelligence development has faced a persistent bottleneck. Traditional computing systems processed information sequentially, much like reading a single line of text word by word, which severely limited how fast massive computing hardware could train these models. `(background)`
- *spoken_text*: Recurrent neural networks limit parallelization due to inherent sequential computation. `[slide 2]`
- *spoken_text*: That operational limit creates significant challenges for national technology infrastructure, energy consumption, and long-term economic competitiveness. `(background)`
- *spoken_text*: A major shift occurred when researchers introduced the Transformer architecture, a breakthrough design that completely replaces traditional recurrent and convolutional layers with attention mechanisms. `[slide 2]`
- *spoken_text*: By allowing computer systems to process entire data sequences simultaneously rather than step by step, this innovation unlocks unprecedented parallelization across modern processor hardware. `[slide 2]`
- *spoken_text*: For policymakers concerned with computational resource allocation, national security capabilities, and public sector technological sovereignty, understanding these efficiency gains is vital. `(framing)`
- *spoken_text*: The new architecture achieved state-of-the-art translation performance on major international benchmarks while drastically reducing training durations. For instance, top-tier models completed intensive training runs in just three and a half days. `[slide 4, slide 6]`
- *spoken_text*: As governments and industries navigate the future of digital infrastructure, supporting architectures that maximize computational efficiency will be vital for sustainable, high-impact innovation. `(background)`

Checks: claims supported: 4 of 4; 90-second: 188 spoken words (about 75 seconds) (missed); requirement "for policymakers": met; requirement "include speaker directions or cues": met

*first words 1.4 s · done in 35.5 s*

---

**User:** Now a 30-second plain-English version for a general audience.

*Tools: write_deliverable*

**SARAL:** I am writing a 30-second plain-English version for a general audience.

I have written a 30-second plain-English script for a general audience. It runs about 26 seconds, meeting your 30-second limit, and all requirements were successfully met. Would you like to review either script or make any adjustments?

*Draft version 1 · status: succeeded*

**#1 30-Second Plain-English Script**

- *hook*: Have you ever wondered how modern artificial intelligence processes language so quickly? `(framing)`
- *context*: Traditional computer systems used to read text one single word at a time, creating a massive bottleneck. `(background)`
- *core_finding*: Researchers recently introduced the Transformer architecture, a breakthrough design that completely replaces old step-by-step methods with attention mechanisms. `[slide 2]`
- *takeaway*: By analyzing entire sentences all at once, this innovation unlocks unprecedented computing speed and powers today's smartest translation tools. `[slide 2]`

Checks: claims supported: 2 of 2; 30-second: 66 spoken words (about 26 seconds) (met); requirement "plain-English": met; requirement "for a general audience": met

*first words 1.4 s · done in 9.9 s*

---

**User:** Write a 5-minute technical script for graduate students.

*Tools: write_deliverable*

**SARAL:** I am writing a 5-minute technical script for graduate students.

I have written the technical script for graduate students. It runs about 241 seconds, short of the 5-minute (300 seconds) limit you asked for `[slide 2, slide 2, slide 3, slide 4]`. Both technical depth and audience requirements were met. Would you like me to expand it so it reaches the full 5-minute duration?

*Draft version 1 · status: succeeded*

**#1 5-Minute Technical Script for Graduate Students on the Transformer Architecture**

- *introduction*: Welcome everyone to today's advanced seminar on deep learning architectures. Today we are examining a foundational shift in sequence modeling: the deliberate transition away from traditional recurrence and convolutions toward purely attention-based mechanisms. `(framing)`
- *context*: To appreciate why this shift represents such a major milestone, let us first review the traditional paradigm. In classical recurrent neural networks, processing text requires sequential computation step by step along the temporal length of the input. `(background)`
- *analysis*: This inherent sequential dependency prevents parallelization across time steps, creating a severe computational bottleneck as sequence lengths grow, because the hidden state at step t depends directly on the hidden state at step t minus one. `[slide 2]`
- *analysis*: Historically, before this architecture was proposed, attention mechanisms were used merely to augment these recurrent structures, helping models focus on relevant parts of the input while still relying fundamentally on recurrence for the underlying sequence backbone. `[slide 2]`
- *core_concept*: The Transformer architecture breaks away from this legacy paradigm entirely by replacing recurrence and convolutions with attention. `[slide 2]`
- *core_concept*: By eliminating recurrent connections completely, this design allows much greater parallelization for sequence models, enabling significantly faster training times across large-scale corpora. `[slide 2]`
- *architecture*: Let us examine the structural mechanics that make this possible. The Transformer adopts a standard encoder-decoder structure for sequence transduction. `[slide 3]`
- *architecture*: Both the encoder and the decoder are constructed from a stack of six identical layers. `[slide 3]`
- *architecture*: Within each of these encoder and decoder layers, you find two core sub-components: multi-head self-attention and a position-wise feed-forward network. `[slide 3]`
- *mechanism*: Crucially, multi-head attention runs eight parallel attention layers, or projections, simultaneously. `[slide 3]`
- *mechanism*: This multi-head design allows the model to jointly attend to information from different representation subspaces at different positions, capturing diverse relational aspects of the text simultaneously. `[slide 5]`
- *mechanism*: Because self-attention is inherently permutation-invariant, the network has no intrinsic notion of token order. To fix this, sinusoidal positional encodings are added to the input embeddings to convey sequence order. `[slide 3]`
- *complexity*: Let us consider the computational complexity implications in detail. Self-attention layers offer substantial computational and operational advantages over recurrent layers. `[slide 5]`
- *complexity*: They require a constant number of sequential operations, whereas recurrent layers require O of n operations. `[slide 5]`
- *complexity*: This reduction in sequential operations permits much greater parallelization and results in significantly shorter maximum path lengths between dependencies in the network, making it easier to learn long-range interactions. `[slide 5]`
- *transition*: How does this theoretical efficiency translate to empirical benchmark performance? The results on standard machine translation tasks are definitive. `(framing)`
- *results*: On the WMT 2014 English-to-German translation task, the big Transformer model achieved a BLEU score of twenty-eight point four. `[slide 4]`
- *results*: This score outperformed all previously published models by over two BLEU points, establishing a new state of the art. `[slide 4]`
- *results*: Similarly, on the WMT 2014 English-to-French translation task, the big model reached a BLEU score of forty-one point eight. `[slide 4]`
- *results*: To achieve these performance levels, training the big model required three point five days on eight P 100 GPUs. `[slide 4]`
- *results*: Furthermore, the base model surpassed all previously published models while requiring significantly less computational cost to train. `[slide 4]`
- *generalization*: Beyond machine translation, the architecture demonstrates strong generalizability across other demanding sequence tasks, such as English constituency parsing. `[slide 5]`
- *conclusion*: In conclusion, by successfully introducing an attention-only sequence transduction model that replaces recurrent layers with multi-headed self-attention, the Transformer establishes a powerful new standard for our field. `[slide 6]`
- *closing*: As graduate students in this domain, it is vital to understand that the architectural principles introduced in this seminal work continue to serve as the foundation for modern large language models. `(background)`
- *closing*: When we analyze how these models scale, the removal of sequential bottlenecks is what enables billions of parameters to be trained efficiently on massive clusters. `(background)`
- *closing*: Thank you for your attention, and I look forward to our discussion on how these self-attention mechanisms might evolve in future architectures. `(framing)`

Checks: claims supported: 20 of 20; 5-minute: 603 spoken words (about 241 seconds) (missed); requirement "technical": met; requirement "for graduate students": met

*first words 1.8 s · done in 31.5 s*

---

**User:** Write a press release about the main result.

*Tools: write_deliverable*

**SARAL:** I am writing a press release about the main result.

I have written a press release about the main result, and all requirements were successfully met. Would you like to review or adjust any of the deliverables we've created so far?

*Draft version 1 · status: succeeded*

**#1 FOR IMMEDIATE RELEASE: New Artificial Intelligence Architecture Breakthrough Shatters Language Translation Benchmarks**

- *lead_paragraph*: Researchers introduce a revolutionary deep learning architecture called the Transformer, which replaces traditional neural network recurrence with an attention-only design to dramatically accelerate training and set new performance standards in machine translation. `[slide 1, slide 2, slide 6]`
- *body_paragraph*: In natural language processing, machine translation involves teaching computer systems to automatically translate text from one human language to another, a critical capability for global communication and digital services. `(background)`
- *body_paragraph*: Traditional sequence models have long relied on recurrent neural networks that process text step-by-step in a sequential chain. This sequential dependency prevents parallel computation across time steps, creating a severe bottleneck when training on massive datasets. `[slide 2]`
- *body_paragraph*: The newly unveiled Transformer architecture completely eliminates recurrence and convolutions, relying entirely on attention mechanisms to connect words and meanings across input sequences. `[slide 2]`
- *body_paragraph*: By removing these sequential bottlenecks, the Transformer enables unprecedented levels of model parallelization, drastically reducing the time required to train advanced sequence models. `[slide 2]`
- *body_paragraph*: To rigorously evaluate this new design, the research team tested the model on standard benchmark translation tasks, specifically the WMT 2014 English-to-German and English-to-French datasets. `[slide 4]`
- *body_paragraph*: The BLEU score is a widely recognized automated metric used in natural language processing to measure the quality of machine-translated text against human reference translations. `(background)`
- *body_paragraph*: On the WMT 2014 English-to-German translation task, the big Transformer model achieved a stellar BLEU score of 28.4, outperforming all previously published models by over two full BLEU points and establishing a new state of the art. `[slide 4, slide 6]`
- *body_paragraph*: Similarly, on the WMT 2014 English-to-French translation task, the big model reached a remarkable BLEU score of 41.8. `[slide 4]`
- *body_paragraph*: Achieving these groundbreaking results required training the big model for just 3.5 days on a hardware setup consisting of eight P100 GPUs. `[slide 4]`
- *body_paragraph*: In addition to the large configuration, the base Transformer model surpassed all previously published models while demanding significantly less computational cost to train. `[slide 4]`
- *body_paragraph*: Beyond its success in machine translation, the Transformer demonstrated strong generalizability across other complex sequence processing challenges, including English constituency parsing. `[slide 5, slide 6]`
- *conclusion*: By establishing attention-only models as a robust new standard, this research opens exciting new pathways for efficiency and scalability in artificial intelligence research and development. `[slide 6]`

Checks: claims supported: 11 of 11; requirement "press release format": met; requirement "focus on the main result": met

*first words 1.4 s · done in 14.8 s*
