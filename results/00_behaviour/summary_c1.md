## Contrast 1 — possessive agreement (bare frames)

Hindi/Punjabi: Δ = logit(fem possessive) − logit(masc possessive) at the decision token. Correct = Δ>0 for feminine nouns, Δ<0 for masculine.

**hi** (n=320, dropped=0, prefix=' मे')  
- accuracy: 0.562 [0.509, 0.622]  
- mean Δ | fem noun: 0.891 [0.805, 0.971]  
- mean Δ | masc noun: 0.398 [0.315, 0.484]  
- mean P(possessive prefix): 0.007  

  - frame 0: acc=0.600 (n=80) e.g. `यह किताब असल में`  

  - frame 1: acc=0.575 (n=80) e.g. `वह किताब असल में`  

  - frame 2: acc=0.550 (n=80) e.g. `यह किताब तो`  

  - frame 3: acc=0.525 (n=80) e.g. `यह किताब सचमुच`  

**pa** (n=316, dropped=0, prefix=' ਮੇ')  
- accuracy: 0.573 [0.516, 0.627]  
- mean Δ | fem noun: 0.178 [0.120, 0.233]  
- mean Δ | masc noun: -0.000 [-0.053, 0.049]  
- mean P(possessive prefix): 0.047  

  - frame 0: acc=0.620 (n=79) e.g. `ਇਹ ਕਿਤਾਬ ਅਸਲ ਵਿੱਚ`  

  - frame 1: acc=0.557 (n=79) e.g. `ਉਹ ਕਿਤਾਬ ਅਸਲ ਵਿੱਚ`  

  - frame 2: acc=0.506 (n=79) e.g. `ਇਹ ਕਿਤਾਬ ਤਾਂ`  

  - frame 3: acc=0.608 (n=79) e.g. `ਇਹ ਕਿਤਾਬ ਸੱਚਮੁੱਚ`  


Controls (en, ta): log p(form) for translations of Hindi-fem vs Hindi-masc nouns; the difference should be ≈0 (no agreement in these languages).

**en** (n=320): mean log p | fem-transl = -9.946, masc-transl = -10.395, diff = 0.449 [-0.273, 1.174]  

**ta** (n=320): mean log p | fem-transl = -10.656, masc-transl = -10.666, diff = 0.010 [-0.318, 0.334]  

## Contrast 1 — possessive agreement (ctx frames)

Hindi/Punjabi: Δ = logit(fem possessive) − logit(masc possessive) at the decision token. Correct = Δ>0 for feminine nouns, Δ<0 for masculine.

**hi** (n=304, dropped=0, prefix=' मे')  
- accuracy: 0.905 [0.868, 0.934]  
- mean Δ | fem noun: 2.184 [2.053, 2.307]  
- mean Δ | masc noun: -0.981 [-1.134, -0.830]  
- mean P(possessive prefix): 0.521  

  - frame 0: acc=0.934 (n=76) e.g. `यह कलम मेरी है। यह घर मेरा है। यह किताब असल में`  

  - frame 1: acc=0.882 (n=76) e.g. `यह घर मेरा है। यह कलम मेरी है। यह किताब असल में`  

  - frame 2: acc=0.921 (n=76) e.g. `यह कुर्सी मेरी है। यह कमरा मेरा है। यह किताब असल में`  

  - frame 3: acc=0.882 (n=76) e.g. `यह कमरा मेरा है। यह कुर्सी मेरी है। यह किताब असल में`  

**pa** (n=300, dropped=0, prefix=' ਮੇ')  
- accuracy: 0.793 [0.747, 0.840]  
- mean Δ | fem noun: 0.241 [0.109, 0.370]  
- mean Δ | masc noun: -1.241 [-1.406, -1.078]  
- mean P(possessive prefix): 0.359  

  - frame 0: acc=0.760 (n=75) e.g. `ਇਹ ਕਲਮ ਮੇਰੀ ਹੈ। ਇਹ ਘਰ ਮੇਰਾ ਹੈ। ਇਹ ਕਿਤਾਬ ਅਸਲ ਵਿੱਚ`  

  - frame 1: acc=0.773 (n=75) e.g. `ਇਹ ਘਰ ਮੇਰਾ ਹੈ। ਇਹ ਕਲਮ ਮੇਰੀ ਹੈ। ਇਹ ਕਿਤਾਬ ਅਸਲ ਵਿੱਚ`  

  - frame 2: acc=0.787 (n=75) e.g. `ਇਹ ਕੁਰਸੀ ਮੇਰੀ ਹੈ। ਇਹ ਕਮਰਾ ਮੇਰਾ ਹੈ। ਇਹ ਕਿਤਾਬ ਅਸਲ ਵਿੱਚ`  

  - frame 3: acc=0.853 (n=75) e.g. `ਇਹ ਕਮਰਾ ਮੇਰਾ ਹੈ। ਇਹ ਕੁਰਸੀ ਮੇਰੀ ਹੈ। ਇਹ ਕਿਤਾਬ ਅਸਲ ਵਿੱਚ`  


Controls (en, ta): log p(form) for translations of Hindi-fem vs Hindi-masc nouns; the difference should be ≈0 (no agreement in these languages).

**en** (n=304): mean log p | fem-transl = -0.672, masc-transl = -0.630, diff = -0.042 [-0.084, 0.000]  

**ta** (n=304): mean log p | fem-transl = -0.554, masc-transl = -0.509, diff = -0.045 [-0.096, 0.011]  
