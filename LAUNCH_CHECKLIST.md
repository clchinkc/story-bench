# 🚀 Story Theory Benchmark - Launch Checklist

**Status**: Repo is PUBLIC
**Primary Goal**: Get 500+ Reddit upvotes, 1M+ Twitter impressions, answer all comments
**Timezone**: UTC+8

---

## 📱 LAUNCH DAY TIMELINE (UTC+8 times → targets US traffic)

### 9:00 PM UTC+8 - TWITTER LAUNCH 🐦
*Posts hit US morning (8 AM ET / 5 AM PT)*

Post the 8-post Twitter thread (30-60 min apart, each replies to previous):
- [ ] **Post 1** (9:00 PM): Main announcement
- [ ] **Post 2** (9:45 PM): Key findings + image
- [ ] **Post 3** (10:30 PM): Why story theory
- [ ] **Post 4** (11:15 PM): Agentic tasks
- [ ] **Post 5** (12:00 AM): Cost analysis + image
- [ ] **Post 6** (12:45 AM): Scale & rigor
- [ ] **Post 7** (1:30 AM): The challenge + image
- [ ] **Post 8** (2:00 AM): Call to action

### 10:00 PM UTC+8 - REDDIT LAUNCH 🤖
*Posts hit US peak daytime (9 AM ET / 6 AM PT)*

**Strategy**: Post to 5 subreddits (1 every 1-1.5 hours, starting with most technical)

- [ ] **10:00 PM**: r/MachineLearning (REDDIT_POST_MachineLearning.md)
  - Title: `[R] Story Theory Benchmark: Rigorously evaluating LLM narrative generation...`
  - Use research-style format with [R] tag

- [ ] **11:00 PM**: r/WritingWithAI (REDDIT_POST_WritingWithAI.md)
  - Title: `Story Theory Benchmark: Test your AI model's narrative generation...`
  - Practical, tool-focused angle

- [ ] **12:00 AM**: r/AItools (REDDIT_POST_AItools.md)
  - Title: `Story Theory Benchmark: 34-task open-source evaluation framework...`
  - Comparison table, discovery angle

- [ ] **1:00 AM**: r/ArtificialIntelligence (REDDIT_POST_ArtificialIntelligence.md)
  - Title: `Story Theory Benchmark released: Rigorous evaluation showing...`
  - News angle, discussion questions

- [ ] **2:00 AM**: r/LLM (REDDIT_POST_LLM.md)
  - Title: `DeepSeek v3.2 achieves 92% while Claude costs $2.85 for 90.8%...`
  - Cost-benefit analysis, practical recommendations

### 10:30 PM UTC+8 - HACKER NEWS 🟠
- [ ] Submit: https://news.ycombinator.com/submit
  - Title: `Story Theory Benchmark: 34-task evaluation framework for LLM narrative generation`
  - URL: `https://github.com/clchinkc/story-bench`

### DURING LAUNCH: ENGAGEMENT & MONITORING ✍️
- [ ] Check Twitter every 15 min for replies
- [ ] Check Reddit every 30 min for comments
- [ ] **Respond to ALL comments within 2 hours**
- [ ] Thank early supporters by name
- [ ] Answer technical questions patiently
- [ ] Share r/MachineLearning post link on Twitter as bonus tweet
- [ ] Fix any bugs immediately if found

### NEXT MORNING (Day 2)
- [ ] Publish Medium post (MEDIUM_POST.md)
  - Title: `The Verification Problem: Why Evaluating Creative AI Is Harder Than Code`
  - Include benchmark_cost_vs_score.png image
- [ ] Post LinkedIn article
- [ ] DM 3-5 AI influencers (tag in Twitter if comfortable)
- [ ] Compile first day metrics (upvotes, comments, stars)

---

## 📊 KEY STATS TO MENTION

- **34 tasks** across 9 types (5 standard + 4 agentic)
- **21 models** tested
- **5 story frameworks** covered
- **3 evaluators** per generation (median aggregated)
- **714 generations**, **2,142 evaluations**
- **~$36 total compute cost**

## 🏆 QUICK RANKINGS

| Rank | Model | Score | Cost |
|------|-------|-------|------|
| 🥇 | DeepSeek v3.2 | 91.9% | $0.20 |
| 🥈 | Claude Opus 4.5 | 90.8% | $2.85 |
| 🥉 | Claude Sonnet 4.5 | 90.1% | $1.74 |
| 4 | Claude Sonnet-4 | 89.6% | $1.59 |
| 5 | o3 | 89.3% | $0.96 |
| 6 | Gemini 3-flash-preview | 88.3% | $0.59 |

**Full leaderboard**: https://github.com/clchinkc/story-bench/blob/main/results/LEADERBOARD.md

---

## 💬 QUICK RESPONSES (Copy-Paste Ready)

### "Why should I care?"
> Tests narrative **structure understanding**, not pattern matching. 34 tasks, 5 frameworks, rigorous evaluation. Agentic tasks show ~2× larger capability gaps than single-shot.

### "How do I use it?"
```bash
python3 run.py --gen-model "your/model-id"
python3 visualize.py  # see results
```

### "What makes it different?"
> Agentic tasks show ~2× larger capability gaps than single-shot tasks. Exposes differences in iterative reasoning.

### "Is my model good?"
> Compare to leaderboard: https://github.com/clchinkc/story-bench/results/LEADERBOARD.md

### "How do I add my model?"
> 1. Update `config/models.yaml`
> 2. Run benchmark: `python3 run.py --gen-model "your/model-id"`
> 3. Submit PR with results

### "Can I use it commercially?"
> Yes! MIT license.

### "How much does it cost?"
> ~$0.05 per generation when you include evaluation. Total ~$35 for all 20 models.

---

## 🔗 KEY LINKS (All Pre-formatted)

```
GitHub Repo:
https://github.com/clchinkc/story-bench

Leaderboard:
https://github.com/clchinkc/story-bench/blob/main/results/LEADERBOARD.md

Task Analysis:
https://github.com/clchinkc/story-bench/blob/main/results/TASK_ANALYSIS.md

Full Documentation:
https://github.com/clchinkc/story-bench/blob/main/CLAUDE.md

Twitter:
@firstoryapp

Medium Post:
MEDIUM_POST.md (publish to Medium)
```

---

## 🔄 CTA FLOW (Cross-Platform Links)

```
Medium → GitHub (detailed results, run yourself)
       → Twitter (follow for updates)

Twitter → GitHub (star the repo)

Reddit → GitHub (detailed methodology, run yourself)
       → Medium (full analysis post)
       → Twitter (follow for updates)
```

**Consistency**: All platforms point to GitHub as the primary destination. Medium provides deeper analysis, Twitter for ongoing updates.

---

## 🚨 IF SOMETHING GOES WRONG

### Website/Links broken
- Fix immediately
- Post: "Found a bug in link. Fixed now. Thanks for catching!"
- No shame in mistakes - show responsiveness

### Results questioned
- Share the evaluation code
- Say: "Great question! Here's exactly how we [...check CLAUDE.md...]"
- Offer to re-run with their specs

### Model team complains score too low
- Be gracious: "Thanks for feedback. Here's the task/eval code."
- "If you see an issue with our evaluation, we'd love a PR."
- Don't get defensive

---

## 🎯 SUCCESS METRICS

**Day 1 Targets** (aim for at least one):
- 500+ upvotes across all Reddit posts
- 1M+ Twitter impressions
- 10+ positive comments from researchers
- 0 major bugs found
- Posted on all 5 subreddits successfully

**Week 1 Targets**:
- 1K-2K GitHub stars
- 2-3M Twitter impressions
- 10-20 model submissions
- First community contributions

---

## 💡 REMEMBER

- Researchers love reproducible evaluations
- People respect transparency & humility
- First impressions matter (respond quickly)
- Community grows over time (not overnight)
- You've built something solid → let it shine
- **You're an ambassador, not a salesperson** - answer genuinely, admit limitations, invite feedback

---

## 📞 YOUR ROLE DURING LAUNCH

**You are**: Ambassador, not salesperson
- Answer questions genuinely
- Admit limitations ("Good catch!")
- Invite feedback ("How would you improve it?")
- Celebrate community ("Amazing contribution!")
- Be patient and humble
- Show enthusiasm but stay professional

---

## ✨ AFTER LAUNCH

### Week 2-3:
- Monitor Reddit discussions
- Respond to GitHub issues
- Feature community contributions
- Check overnight social metrics

### Month 1:
- Blog post: "Why 90% of LLMs fail at multi-beat synthesis"
- Consider video walkthrough
- Engage with press coverage

---

**You've got this! 🚀**
