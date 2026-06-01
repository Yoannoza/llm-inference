# infer-metrics

> Un outil minimal et reproductible pour mesurer **les métriques-mères de l'inférence LLM** sur n'importe quelle API compatible OpenAI.

**Statut : v1.0 — les 4 familles sont en place (Latence, Débit, Coût, Ressources) + rapport unifié JSON.** Construit en public.

---

## Pourquoi cet outil

Le coût de l'inférence est devenu le vrai problème économique de l'IA. Mais quand on veut comprendre les performances d'un modèle, on tombe soit sur des articles très théoriques, soit sur du code brut sans explication. Cet outil est le chaînon manquant : **un harnais minimal qu'on peut lancer en 30 secondes** sur GPT-4o, Claude, un modèle Together, Groq, ou son propre vLLM local — pour voir de ses propres yeux ce que valent TTFT et TPOT.

C'est aussi le premier pas d'une ressource pédagogique plus large que je construis en public : un guide complet sur l'évaluation de l'inférence, théorie + pratique reproductible.

## Le modèle mental : 4 familles de métriques

Notre découpage synthétise les principales sources (draft IETF [`draft-gaikwad-llm-benchmarking-terminology`](https://datatracker.ietf.org/doc/html/draft-gaikwad-llm-benchmarking-terminology), BentoML LLM Inference Handbook, NVIDIA GenAI-Perf). Chaque famille a une « métrique-mère » que l'outil cible :

| Famille | Métrique-mère | Statut |
|---|---|---|
| **Latence** | TTFT (Time To First Token) + TPOT (Time Per Output Token) | ✅ v0.1 |
| **Débit** | Output tokens / second (système) | ✅ v0.2 |
| **Coût** | $ / 1M tokens de sortie | ✅ v0.3 |
| **Ressources** | VRAM utilisée (local uniquement) | ✅ v0.4 |
| **Rapport unifié** | JSON + résumé texte | ✅ v1.0 |

> Note : le draft IETF reconnaît officiellement 4 familles principales (latence, débit, ressources, **qualité**). On a délibérément retiré la qualité de cet outil pour rester simple — la qualité demande un dataset d'évaluation et c'est un sujet à part entière.

## Installation

```bash
git clone https://github.com/<ton-user>/infer-metrics.git
cd infer-metrics
pip install -r requirements.txt
```

## Usage en 30 secondes

L'outil expose **5 sous-commandes** (une par famille + le rapport unifié) :

```bash
# v0.1 — Latence (TTFT + TPOT)
python -m src.cli latency \
  --base-url https://api.openai.com/v1 --model gpt-4o-mini --runs 5

# v0.2 — Débit : courbe latence/débit par niveau de concurrence
python -m src.cli throughput \
  --base-url https://api.openai.com/v1 --model gpt-4o-mini \
  --concurrencies 1,2,4,8 --requests 8

# v0.3 — Coût : à partir d'un vrai run, ou en analytique
python -m src.cli cost --model gpt-4o-mini --from-run \
  --base-url https://api.openai.com/v1
python -m src.cli cost --model gpt-4o-mini \
  --input-tokens 500 --output-tokens 1500

# v0.4 — Ressources : VRAM peak (requiert nvidia-smi sur le hôte du modèle)
python -m src.cli resources \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-3B-Instruct \
  --runs 3 --gpu-index 0

# v1.0 — Rapport unifié (toutes les familles) + JSON archivable
python -m src.cli report \
  --base-url https://api.openai.com/v1 --model gpt-4o-mini \
  --runs 5 --concurrencies 1,4 --json-out report.json
```

### Dashboard HTML

Ouvre `dashboard.html` dans un navigateur (double-clic suffit, aucun serveur, zéro dépendance). Une démo s'affiche par défaut — glisse-dépose ton `report.json` n'importe où sur la page pour voir ton propre relevé. Thème instrument de mesure, prêt pour les captures d'écran.

**Sur Groq, Together, Fireworks, DeepSeek, vLLM local, etc.** — même commandes, change juste `--base-url` et `--api-key`. Tout ce qui parle OpenAI-compatible marche.

### Sortie attendue (latence)

```
→ Modèle : gpt-4o-mini
→ Endpoint : https://api.openai.com/v1
→ 1 warmup + 5 runs mesurés

  [warmup 1/1] ... TTFT=  342.1 ms | TPOT=   24.3 ms/tok | total= 3450.2 ms | tokens=128
  [run     1/5] ... TTFT=  198.4 ms | TPOT=   23.1 ms/tok | total= 3134.7 ms | tokens=128
  ...

════════════════════════════════════════════════════════════
  LATENCE — résumé sur 5 runs (médianes)
════════════════════════════════════════════════════════════
  TTFT médian       :   210.5 ms   (min 198 / max 245)
  TPOT médian       :    23.8 ms/token
  → débit observé   :    42.0 tokens/s
════════════════════════════════════════════════════════════
```

### Sortie attendue (rapport unifié)

```
── LATENCY ──
  [run 1/5] TTFT=  198 ms | TPOT=  23.1 ms/tok | total= 3134 ms | tokens=128
  ...
── THROUGHPUT ──
  conc=  1 | débit=   42.1 tok/s | TTFT_med=  205 ms | TPOT_med= 23.7 ms/tok
  conc=  4 | débit=  148.6 tok/s | TTFT_med=  410 ms | TPOT_med= 24.9 ms/tok
── COST ──
── RESOURCES ──  (skipped sur Mac / API distante)

════════════════════════════════════════════════════════════
  RAPPORT UNIFIÉ — gpt-4o-mini
════════════════════════════════════════════════════════════
  Latence   : TTFT 205 ms  |  TPOT 23.7 ms/tok
  Débit pic : 148.6 tok/s @ conc 4
  Coût      : $0.60 / 1M tokens output
  VRAM      : (skipped — nvidia-smi introuvable)
════════════════════════════════════════════════════════════
```

## Définitions

Alignées sur le draft IETF en cours de standardisation :

- **TTFT** : délai entre l'envoi de la requête et la réception du **premier** token de sortie. Dominé par le *prefill* (lecture du prompt) et la file d'attente. C'est la métrique critique pour le ressenti utilisateur.
- **TPOT** : temps moyen pour chaque token **après** le premier. Calculé : `(latence_totale - TTFT) / (n_tokens - 1)`. Reflète l'efficacité du *decode*. C'est la « vitesse de frappe » perçue.

## Limites assumées

- **Mesure côté client.** Inclut le temps réseau. Pour isoler le serveur, lancer depuis la même machine.
- **Comptage de tokens côté client.** L'outil compte les chunks reçus, pas les tokens du tokenizer du modèle. Pour la plupart des APIs OpenAI-compat, 1 chunk ≈ 1 token, mais ce n'est pas garanti.
- **Pas d'analyse de percentiles (P95, P99).** Pour ça il faut au moins 1000 runs (cf. §5.3 du draft IETF). À venir dans une version « charge ».

## Roadmap

- [x] **v0.1** — Latence (TTFT + TPOT)
- [x] **v0.2** — Débit (sweep de concurrence, courbe latence/débit système)
- [x] **v0.3** — Coût (catalogue de tarifs publics + override CLI)
- [x] **v0.4** — Ressources (VRAM peak via polling nvidia-smi)
- [x] **v1.0** — Rapport unifié (texte + JSON archivable)
- [ ] **v1.1** — Notebook d'analyse comparée + percentiles (P95/P99) sous charge

## Références

- [IETF Draft — Gaikwad, *Benchmarking Terminology for LLM Serving*, Jan 2026](https://datatracker.ietf.org/doc/html/draft-gaikwad-llm-benchmarking-terminology) *(individual draft, en cours)*
- [BentoML — LLM Inference Handbook](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)
- [DigitalOcean — LLM Inference Benchmarking, Feb 2026](https://www.digitalocean.com/blog/llm-inference-benchmarking)

## Licence

MIT.
