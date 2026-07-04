# Note per il meeting (in italiano) — aggiornamenti dal feedback del 27 giugno

Punto di partenza: train loss ~0.05, val loss ~0.5, CSI@0.05 bloccato intorno a 0.6,
tutto sull'evento dell'Ahr 2021 (una sola simulazione, warm start, riferimento SFINCS a 100 m).

---

## 1. "Prova MAE al posto del peso per le acque basse" → prima ho fatto una diagnosi

**L'idea dietro il suggerimento.** Con la RMSE gli errori vengono elevati al quadrato, quindi i
pochi errori grandi (celle profonde, il fiume) dominano la loss e il modello può trascurare i tanti
errori piccoli nelle celle poco profonde — che però sono proprio quelle che decidono se una cella è
"bagnata o asciutta", cioè il CSI. Ci sono due modi per correggere questo squilibrio: dare
esplicitamente più peso alle celle basse (shallow_weight, quello che usiamo ora) oppure passare alla
MAE, dove gli errori contano in modo lineare e quindi quelli piccoli pesano relativamente di più.
Sono due strade per lo stesso obiettivo.

**Cosa ho fatto.** Prima di cambiare la loss ho verificato se il problema del CSI è davvero questo.
Ho costruito un notebook diagnostico che confronta 4 modelli sullo stesso rollout completo di 118 ore
e scompone gli errori di classificazione.

**Cosa è emerso:**
- Il CSI **decade in modo monotono** da 1.0 (warm start) fino a ~0.35–0.45: è accumulo di errore
  lungo il rollout (drift), non un problema di loss.
- Dominano i **falsi allarmi**: il rapporto FP/celle-bagnate cresce fino a 0.7–1.3 a fine rollout —
  il modello bagna sempre più celle e non si "asciuga" mai nella fase di recessione.
- Nella fase di crescita della piena c'è una **banda compatta di celle mancate a valle**: l'onda di
  piena del modello arriva in ritardo rispetto a SFINCS (problema di tempistica del fronte).
- **L'ipotesi "errori vicino alla soglia" è smentita**: solo l'1–14 % delle celle mancate ha
  profondità reale entro 5 cm dalla soglia di 0.05 m. Le celle mancate sono profonde (il fronte in
  ritardo), e anche i falsi allarmi prevedono profondità oltre 0.5 m.

**Conclusione.** Gli errori che bloccano il CSI sono grandi e spazialmente coerenti (drift + ritardo
del fronte), non piccoli errori trascurati nelle celle basse. Quindi né la MAE né più shallow_weight
colpiscono il problema vero — anzi, la MAE *ridurrebbe* la penalità proprio sugli errori grandi che
dominano. Conferma empirica: le run con shallow_weight 3 e 5 finiscono nella stessa fascia di CSI.
(Se vogliono comunque il controllo empirico: una run MAE è economica, si può lanciare in una notte.)

**Cosa mostrare:**
- `utils/compare_weighted_rmse_runs.ipynb`:
  - il grafico "CSI @ 0.05 m over time" (decadimento monotono, tutte le run);
  - il pannello a 3 grafici "wet area / FP / FN nel tempo" (i falsi allarmi crescono sempre);
  - le mappe spaziali TP/FP/FN a rising limb / picco / recessione (banda arancione = fronte in
    ritardo; rosso sparso = sovra-bagnamento in recessione);
  - gli istogrammi "fringe analysis" con la tabella (% errori vicino alla soglia: bassa).
- (Opzionale, codice) `training/loss.py`, funzione `loss_function`: si vede shallow_weight (√w sul
  residuo) e velocity_scaler — utile se chiedono come è fatta la loss adesso.

---

## 2. "Aggiungi le ghost cells di outflow" → fatto, e ho scoperto due cose importanti

**Cosa sono e cosa ho costruito.** Le ghost cells sono celle finte appena fuori dal bordo del
dominio, speculari alle celle di bordo reali: danno al bordo un posto dove "mandare" l'acqua, come
nei solutori numerici. Il nostro dominio SFINCS ha 87 celle di uscita a valle (nord-est, msk==3).
Ho creato 87 ghost cells speculari, senza valore imposto (il modello impara che restano asciutte).
Il dataset è stato ricostruito e i training sono girati su HAL8.

**Scoperta (a): la condizione di monte "barava".** Validando il nuovo dataset ho scoperto che al
modello non veniva data la portata Q (l'idrogramma da sfincs.dis), ma la **profondità d'acqua
calcolata da SFINCS stesso** nei 7 punti sorgente, a ogni passo. In pratica gli passavamo un pezzo
della risposta (leakage). Succedeva per un ramo di fallback nello script di conversione: il modello
SFINCS non ha celle msk==2 (l'acqua entra da sorgenti puntuali .src), quindi lo script ripiegava
sulla profondità reale. **Corretto**: ora il modello riceve la vera portata Q (type_BC=2, picco
450 m³/s), stessa convenzione degli altri dataset. Verificato sia in locale sia su HAL8.

**Scoperta (b): il modello ignora completamente il forcing (risultato chiave).** Ho fatto un test di
ablazione: stessa simulazione di 118 ore, due volte — una con l'idrogramma vero, una con il forcing
**azzerato** (nessuna acqua in ingresso, per quanto ne sa il modello).

| | idrogramma vero | forcing a zero |
|---|---|---|
| CSI@0.05 | 0.4745 | 0.4753 |

Identico. Il modello produce la stessa alluvione in entrambi i casi: addestrato su un solo evento,
ha **memorizzato** "dopo un warm start così, l'Ahr si allaga con questi tempi". L'input di portata è
decorativo. Conseguenza: nessuna modifica di loss o training più lungo su un solo evento darà un
modello che risponde a idrogrammi diversi — servono dati con **idrogrammi diversi**.

**Scoperta (c): le ghost cells di outflow, così come sono, non possono fare nulla.** Gli archi del
grafo che le collegano puntano solo verso l'esterno (interno → ghost). Nella nostra GNN
l'informazione viaggia nella direzione degli archi, quindi le ghost cells *ricevono* ma non
*inviano* mai nulla: nessuna cella interna "sa" che esistono. Sono di fatto inerti (87 predizioni in
più con target ~0, zero effetto sulla previsione). La soluzione — bordo "assorbente": archi
bidirezionali + ghost forzate asciutte, così le celle interne vedono un vicino sempre asciutto verso
cui drenare — è specificata ma rimandata; aiuterebbe proprio il sovra-bagnamento in recessione visto
al punto 1.

**Cosa mostrare:**
- `database/create_dataset_inflow_outflow_gc.ipynb`, Step 4: la mappa finale delle BC — 7 nodi di
  inflow con Q (triangoli rossi, con i nomi delle sorgenti: Kreuzberg, Denn, Kirmutscheid, Müsch,
  new1, Niederadenau, new2) + 87 ghost cells di outflow a valle (triangoli blu).
- La tabella dell'ablazione qui sopra (è la slide più importante del meeting).
- (Opzionale, codice) `run_convert_warmstart_inflow_outflow_gc.py`: la docstring/il blocco BC — si
  vede che ora costruisce la BC da sfincs.src + sfincs.dis (Q, type_BC=2); si può raccontare che
  prima c'era il fallback con la WD reale (type_BC=1).
- (Opzionale, codice) `models/gnn.py`, riga ~438 `scatter(shift_sum, col, ...)`: i messaggi vanno
  da riga a colonna dell'edge_index → con archi solo interno→ghost, l'interno non riceve mai nulla.

---

## 3. "Allena più a lungo" → fatto, con early stopping

**Cosa ho fatto.** Ho rilanciato i training con il forcing corretto (Q): 2000 epoche massime,
early stopping con patience 500, scheduler cosine. Tre run su HAL8:

| run | CSI@0.05 | CSI@0.30 | RMSE rollout WD |
|---|---|---|---|
| 233531 | **0.593** | 0.612 | 0.257 |
| 233342 | 0.567 | 0.611 | 0.297 |
| 233547 | 0.533 | 0.588 | 0.259 |

**Risultato.** Stessa fascia di tutte le run precedenti su singolo evento (0.53–0.67). Togliere il
leakage non è costato nulla — coerente con l'ablazione (tanto il forcing non veniva usato).

**Attenzione a leggere W&B.** Le curve delle nuove run *sembrano* peggiori, per due motivi che non
c'entrano con la qualità del modello: (1) lo scheduler ora ha T_max=2000, quindi a parità di epoca il
learning rate è molto più alto → curve più rumorose e convergenza spostata in avanti; (2) la run
migliore sul grafico (blu, `hid64_weighted_RMSE_loss0_coslr`) ha 700k parametri contro 190k e usa un
curriculum fino a rollout 10. Inoltre la **train_loss non è confrontabile tra run** con
shallow_weight diverso (3 vs 5): a parità di errore fisico il numero è diverso. Confrontabili sono
val_loss e val_CSI (rollout completo, senza pesi).

**Cosa mostrare:**
- Screenshot W&B: pannello val_CSI_005 (e val_loss in scala log, come suggerito nel feedback),
  evidenziando le due run nuove (verde chiaro e rosa chiaro) rispetto alle hid32 comparabili.
- La tabella qui sopra con i numeri finali dai log (`grep "test CSI" logs/...`).

---

## 4. Conclusione e prossimi passi proposti

**Conclusione generale.** Cambiando loss, forcing, ghost cells, epoche e scheduler, il training su
un singolo evento si ferma sempre a CSI@0.05 ≈ 0.53–0.67: è un **soffitto di memorizzazione**.
Onestà importante da dire: tutti i CSI attuali sono calcolati **sull'evento di training** (il codice
stesso avvisa "the validation dataset you are using is the training one"), quindi misurano in parte
memorizzazione, non capacità di generalizzare. Il numero che serve alla tesi è il CSI su un
idrogramma mai visto.

**Prossimi passi, in ordine di priorità:**
1. **Generare nuove run SFINCS sullo stesso dominio con idrogrammi scalati** (es. 0.5× / 0.75× /
   1.25×). È l'unica modifica che: (a) costringe il modello a usare davvero la portata in ingresso,
   (b) permette la valutazione su un evento tenuto fuori dal training, (c) serve direttamente
   all'obiettivo finale degli scenari what-if.
2. **Test economico in parallelo**: aumentare rollout_steps oltre 5 nel training (la run migliore su
   W&B usa un curriculum fino a 10) — attacca direttamente il drift che domina il decadimento del CSI.
3. Più avanti / opzionali: termine soft-CSI (Dice) nella loss per sopprimere i falsi allarmi sparsi;
   bordo di outflow assorbente (punto 2c) per aiutare l'asciugatura in recessione.

**Cosa mostrare:**
- Di nuovo il grafico "CSI over time" (il decadimento è l'argomento visivo per il punto 2 dei
  prossimi passi) e la tabella dell'ablazione (l'argomento per il punto 1).
