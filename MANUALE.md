# frex — Manuale operativo

Guida pratica all'uso quotidiano. Per i dettagli di progetto vedi `README.md`.

---

## 1. Installazione (una volta sola)

Serve Python 3.10+ e `ffmpeg` installato sul sistema.

```bash
git clone https://github.com/Kocciacus/face-recognition-exporter.git
cd face-recognition-exporter
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# CPU
pip install -e ".[recognition]"

# oppure GPU NVIDIA (molto più veloce sui video)
pip install -e ".[gpu]"
```

Al primo avvio InsightFace scarica da solo il modello `buffalo_l` (~300 MB) in `~/.insightface/models`. Serve connessione internet solo quella volta.

Verifica:

```bash
frex --help
```

---

## 2. Il flusso in 4 comandi

| Comando | Cosa fa |
|---|---|
| `frex run CARTELLA` | scansiona, analizza, crea/riusa le cartelle e sposta i file |
| `frex status CARTELLA` | riepilogo: quanti file per stato, quante identità |
| `frex apply-names CARTELLA` | applica solo i rinomini di `folder_names.csv`, senza analizzare nulla |
| `frex export CARTELLA` | rigenera i due CSV dal database |

`CARTELLA` è la radice dell'archivio: tutto (database, CSV, sottocartelle delle persone) vive lì dentro.

---

## 3. Primo utilizzo consigliato

Non lanciarlo subito su tutto l'archivio. Procedura consigliata:

```bash
# 1. prova a vuoto su un campione: decide ma non sposta niente
frex run /media/archivio --dry-run --limit 100

# 2. guarda cosa avrebbe fatto
cat /media/archivio/face_index.csv

# 3. se il risultato convince, esegui davvero sullo stesso campione
frex run /media/archivio --limit 100

# 4. via libera a tutto l'archivio
frex run /media/archivio
```

Puoi interrompere con `Ctrl+C` in qualsiasi momento: al rilancio riprende dai file non ancora processati, non rianalizza quelli già fatti.

---

## 4. File generati

Dentro la cartella dell'archivio:

```
/media/archivio/
├── face_index.csv        <- l'indice leggibile: file -> destinazione
├── folder_names.csv      <- qui scrivi tu i nomi che vuoi per le cartelle
├── person_0001/          <- cartelle create dal programma (rinominabili)
│   ├── .identity         <- marcatore: NON cancellarlo
│   └── video1.mp4
├── person_0002/
└── .face_index/          <- stato interno: NON cancellare
    ├── state.sqlite      <- database (fonte di verità: embedding, storico)
    └── thumbnails/       <- anteprima del volto di ogni identità
```

Regola d'oro: **`.face_index/` e i file `.identity` sono la memoria del programma.** Se li cancelli, alla prossima esecuzione tutte le persone risultano sconosciute e vengono ricreate cartelle nuove.

I CSV invece sono rigenerabili in qualsiasi momento con `frex export`.

---

## 5. Leggere `face_index.csv`

```csv
file,destination,status,identity_id,similarity,faces_found,note
Barack Obama/obama_clip1.mp4,Barack Obama,DONE,56ccf7a6d4dc,,2,
Barack Obama/obama_clip2.mp4,Barack Obama,DONE,56ccf7a6d4dc,0.779,2,
landscape.mp4,E,NO_FACE,,,0,no usable face
```

- **file**: percorso attuale, relativo alla radice dell'archivio
- **destination**: cartella in cui è finito, oppure un marcatore di errore
- **similarity**: quanto somigliava all'identità già nota (1.00 = identico). Vuoto quando l'identità è stata creata proprio con quel file
- **faces_found**: volti utilizzabili trovati nel file

Valori di `destination` / `status`:

| destination | status | Significato | Cosa fare |
|---|---|---|---|
| nome cartella | `DONE` | assegnato | niente |
| `E` | `NO_FACE` | nessun volto riconoscibile | il file resta dov'è; controlla a mano |
| `U` | `UNCERTAIN` | somiglianza in zona grigia | il file resta dov'è; decidi tu |
| `F` | `FAILED` | file illeggibile o codec non supportato | verifica il file |
| `M` | `MISSING` | era indicizzato ma non è più sul disco | niente, è solo storico |
| vuoto | `PENDING` | non ancora analizzato | rilancia `frex run` |

Per rimettere in coda i file `E`, `U` e `F` (per esempio dopo aver abbassato le soglie):

```bash
frex run /media/archivio --retry-failed
```

---

## 6. Dare un nome alle cartelle

Due modi, entrambi validi e combinabili.

### A. Rinomina dal CSV (consigliato)

Apri `folder_names.csv` e compila **solo** la colonna `desired_name`:

```csv
identity_id,current_folder,desired_name,files,observations
56ccf7a6d4dc,person_0001,Nonna Maria,42,118
6f05ab9d3b0c,person_0002,Zio Piero,17,55
```

Poi:

```bash
frex apply-names /media/archivio
```

La cartella viene rinominata e le destinazioni nel database e nel CSV vengono aggiornate. Lo stesso avviene automaticamente anche all'inizio di ogni `frex run`.

### B. Rinomina a mano nel file manager

Puoi semplicemente rinominare la cartella dal tuo sistema operativo: grazie al file `.identity` contenuto al suo interno, il programma la ritrova comunque e adotta il nuovo nome alla prossima esecuzione.

**Da non fare:** cancellare il file `.identity`, o spostare i file di una persona in una cartella creata a mano senza `.identity` (per il programma diventa una cartella qualsiasi e i file verranno ripresi e rismistati).

Se sposti un file già assegnato di nuovo nella radice, la prossima esecuzione lo rimette nella sua cartella senza rianalizzarlo.

---

## 7. Come decide il programma

- **Un solo soggetto noto** → il file va nella sua cartella.
- **Più soggetti, almeno uno noto** → il file va nella cartella del soggetto noto più affidabile (per somiglianza e rilevanza nel video); gli altri soggetti vengono ignorati e non generano cartelle.
- **Nessun soggetto noto** → viene creata **una sola** nuova cartella, per il soggetto dominante del file.
- **Nessun volto utilizzabile** → `E`, il file non viene spostato.

Nessun file viene mai cancellato o sovrascritto: in caso di nomi uguali la destinazione viene resa univoca con un suffisso.

---

## 8. Opzioni di `frex run`

| Opzione | Default | Quando usarla |
|---|---|---|
| `--dry-run` | — | decide e scrive i CSV ma non sposta nulla |
| `--limit N` | — | ferma dopo N file: ideale per campionare |
| `--device cuda` | `auto` | forza la GPU (di default la rileva da sola) |
| `--workers N` | `2` | più processi in parallelo; alzalo se hai CPU libera |
| `--sample-interval S` | `2.0` | secondi tra un frame campionato e l'altro: `5` è molto più veloce |
| `--max-frames N` | `120` | tetto di frame analizzati per video |
| `--keyframes-only` | — | decodifica solo i keyframe: la modalità più veloce, un filo meno accurata |
| `--match-threshold X` | `0.50` | somiglianza minima per dire "è lui". Più alto = più prudente |
| `--review-threshold X` | `0.38` | sotto `match` ma sopra questa → `U` invece di indovinare |
| `--cluster-threshold X` | `0.55` | quanto raggruppare i volti dentro lo stesso file |
| `--retry-failed` | — | rianalizza i file marcati `E`, `U`, `F` |
| `-v` | — | log di debug |

### Tarare le soglie

- Ti ritrovi **la stessa persona in due cartelle**? Abbassa `--match-threshold` (es. `0.45`).
- Ti ritrovi **due persone diverse nella stessa cartella**? Alzalo (es. `0.55`) e riparti da un campione pulito.
- Cambiare le soglie **non rianalizza** i file già `DONE`: agisce sulle decisioni future e, con `--retry-failed`, sui file rimasti indietro.

---

## 9. Tempi attesi

Il collo di bottiglia è la decodifica video, non il riconoscimento.

Per ~1 TB di video, con i default:

- **GPU NVIDIA**: circa 1-2 giorni
- **solo CPU**: circa 1-2 settimane

Per andare più veloce: `--sample-interval 5`, oppure `--keyframes-only`, oppure alza `--workers`. Puoi spezzare il lavoro in più sessioni: lo stato è persistente.

---

## 10. Problemi frequenti

**"not a directory"** → il percorso passato non esiste o non è una cartella.

**Il download del modello fallisce** → serve internet al primo avvio; controlla proxy/firewall, poi riprova.

**Tanti `F` (FAILED)** → codec mancante. Verifica che `ffmpeg` sia installato e che apra il file: `ffmpeg -i file.mp4`. Le foto HEIC/HEIF richiedono un OpenCV con il relativo supporto.

**Tutto molto lento su CPU** → è previsto; usa la GPU oppure `--keyframes-only`.

**Ho cancellato `.face_index/`** → la memoria è persa. I file già ordinati restano dove sono, ma le persone verranno riconosciute come nuove: conviene rilanciare `frex run` sulla radice e ricostruire.

**Interrotto durante uno spostamento** → nessun problema: l'operazione è registrata e viene completata o annullata al rilancio successivo.
