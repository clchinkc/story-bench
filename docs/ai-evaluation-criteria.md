# AI Engine Evaluation Criteria for Creative Writing

A reference framework for evaluating AI models specifically for fanfiction and creative writing tasks. Complements Story Theory Benchmark's structural evaluation with creative writing quality metrics.

---

## Overview

This framework evaluates AI engines across **7 categories** relevant to creative writing quality. Unlike Story Theory Benchmark's focus on story structure and beat execution, these criteria assess the **human-like quality** and **creative depth** of generated content.

**Use cases**:
- Writers choosing AI assistants for long-form fiction
- Evaluating models for roleplay and character work
- Assessing creative writing capabilities beyond structural correctness

---

## Evaluation Categories

### 1. General Realism (Narrative Logic)

Does the overall narrative make sense? Do events and actions occur logically? Are technical details accurate?

**What to check**:
- Cause-and-effect relationships between scenes
- Character knowledge consistency (what they know vs. don't know)
- Physical/world logic (timing, distances, abilities)
- Technical accuracy in specialized topics

### 2. Emotional Realism (Character Depth)

Do the characters' emotions make sense? Are their reactions nuanced and show depth?

**What to check**:
- Emotional reactions match the situation
- Gradual emotional progression (not instant resolution)
- Subtext and unspoken feelings
- Character-specific emotional patterns (not generic responses)

### 3. Humanity (Natural Writing)

Does the fanfiction sound like it was written by a human? Do instructions integrate seamlessly, or are they visible to readers?

**What to check**:
- Voice consistency throughout
- Natural dialogue (no robotic speech patterns)
- Instructions incorporated into narrative flow
- Avoidance of lists, bullet points, or meta-commentary in output

### 4. Level of Detail

How much detailed description is automatically written for each scenario?

**What to check**:
- Sensory details (sight, sound, smell, touch, taste)
- Environmental storytelling
- Character interiority (thoughts, physical sensations)
- Balance between showing and telling

### 5. Context Window

How many tokens of context does the AI engine have? The more tokens, the better it is at remembering previous chats.

**Impact**:
- Longer story memory = better consistency
- Fewer re-reminding needed
- Ability to reference earlier plot points

### 6. Chat Limits

How many instructions can you post in the chat per set period of time?

**Impact**:
- Iterative editing workflow
- Long conversations vs. session resets
- Creative process flexibility

### 7. Explicitness (Content Policy)

How restrictive are the AI engines in writing NSFW scenes?

**What to check**:
- Willingness to write mature themes
- Handling of violence, romance, sensitive topics
- Flexibility for different creative needs

---

## Model Comparison

| Engine | General Realism | Emotional Realism | Humanity | Detail Level | Context | Chat Limit | Explicitness |
|--------|-----------------|-------------------|----------|--------------|---------|------------|--------------|
| Claude | 9/10 | 9/10 | 9/10 | 9/10 | 190k | ~20-45/5hr | Very restrictive |
| Grok | 8/10 | 8/10 | 8/10 | 8/10 | 128k | ~10/2hr | Very permissive |
| ChatGPT | 8/10 | 8/10 | 8/10 | 8/10 | 60-100k | ~10/5hr | Restrictive |
| DeepSeek | 9/10 | 6/10 | 6/10 | 9/10 | 128k | Near unlimited | Moderately restrictive |
| Gemini | 7/10 | 5/10 | 5/10 | 5/10 | 32k | Unlimited | Very restrictive |

**Summary rankings**:
1. **Claude** - Best overall creative writing quality
2. **Grok** - Best for mature content, good writing
3. **ChatGPT** - Solid balance, widely accessible
4. **DeepSeek** - Best for technical detail, unlimited chat
5. **Gemini** - Limited for creative writing use cases

---

## Detailed Assessments

### Claude

**Strengths**:
- Best performance on core creative metrics
- Realistic narratives with strong emotional handling
- Near-human writing quality
- Excellent detail in descriptions
- Largest context window (190k tokens)

**Weaknesses**:
- Moderate chat limits (20-45 messages per 5 hours)
- Very restrictive content policies

**Best for**: Long-form fiction, emotionally complex stories, detailed worldbuilding

### Grok

**Strengths**:
- Surprising quality for fanfiction writing
- Human-like informal tone
- Good detail level
- Very permissive content policy (will write graphic content)

**Weaknesses**:
- Variable quality between sessions
- Tight chat limits (~10 messages per 2 hours)

**Best for**: Mature content, casual creative writing, unrestricted workflows

### ChatGPT

**Strengths**:
- Strong emotional handling and realistic dialogue
- Good imagery and vivid descriptions
- Solid humanity score (natural writing)
- Widely accessible, familiar interface

**Weaknesses**:
- Most restrictive chat limits
- Very restrictive content policies
- Quality varies by version

**Best for**: General creative writing, dialogue-heavy stories, accessible workflows

### DeepSeek

**Strengths**:
- Excellent general realism and technical detail
- Near-unlimited chat capacity
- Large context window (128k)
- Best for logic-heavy narratives

**Weaknesses**:
- Lower emotional realism and character depth
- More robotic/less human-like writing
- Restrictive content policies

**Best for**: Technical fanfiction (sci-fi, hard magic systems), long conversations

### Gemini

**Strengths**:
- Unlimited chat capacity
- Free and accessible

**Weaknesses**:
- Lowest scores across creative writing metrics
- Dry emotional handling
- Minimal detail
- Smallest context window (32k)
- Very restrictive content policies

**Best for**: Not recommended for creative writing tasks

---

## Relationship to Story Theory Benchmark

This framework complements Story Theory Benchmark's evaluation:

| Story Theory Benchmark | This Framework |
|------------------------|----------------|
| Structural correctness | Creative quality |
| Beat execution | Emotional depth |
| Constraint satisfaction | Natural writing |
| Framework mapping | Character consistency |
| Multi-turn planning | Chat workflow |

**Cross-references**:
- `beat_revision` task → Emotional Realism, Humanity
- `critique_improvement` task → Level of Detail, Emotional Realism
- `agentic_constraint_discovery` → Chat Limit considerations
- All tasks → Context Window impact

**Use together**: Run Story Theory Benchmark for structural scores, use these criteria for creative writing quality assessment.

---

## Limitations and Notes

1. **Free tier focus**: These evaluations are for free versions only. Paid tiers may differ significantly.

2. **Subjective elements**: Humanity, Emotional Realism, and Detail Level are inherently subjective. YMMV.

3. **Rapid changes**: AI capabilities and policies change frequently. Re-evaluate periodically.

4. **Use case dependent**: Best model depends on your specific needs (mature content, emotional depth, etc.).

5. **Not comprehensive**: Other factors like speed, UI, and platform integration matter for practical use.

---

## Sources and Methodology

This framework is based on:
- Hands-on testing with free tiers of each AI engine
- Focused evaluation for fanfiction and creative writing tasks
- User experience metrics (chat limits, context windows)
- Content policy testing for mature themes

---

*Last updated: 2025-12-25*
