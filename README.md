# Monte Carlo Off-Policy

Monte_Carlo_Off_Policy.ipynb é o Notebook Python com o código que foi produzido até agora.

## Esqueleto do Artigo
- o que é off-policy
- explicar o ambiente, e dar dois exemplos de políticas
  - uma epsilon-greedy derivada da política ótima (como pi)
  - e uma totalmente aleatória
- explicar problema de predição da RL
  - não aprende a política
  - só aprende a função V
  - off-policy: aprende V de pi rodando b (contrastar com o on-policy, que aprende V de pi rodando pi)
- explicar o importance sampling (mostrar as duas fórmulas)
- mostrar o (pseudo-)código
- mostrar resultados de um experimento
  - cliff-walking...
  - com ordinary e weighted importance sampling
