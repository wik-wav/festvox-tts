ENGLISH MULTISYN RECORDING SET (CMU ARCTIC, 1132 prompts)
=========================================================
Source: http://festvox.org/cmu_arctic/cmuarctic.data  (public domain,
selected from out-of-copyright Project Gutenberg texts for diphone coverage).

FILES
-----
arctic_reclist.txt   One prompt ID per line (arctic_a0001 ... arctic_b0539).
                     Load in OREMO or recstar as the recording list; each take
                     saves as <ID>.wav (e.g. arctic_a0001.wav).

OREMO-comment.txt    <ID><TAB><sentence to read>, one per line. Put it in the
                     OREMO result/destination folder so the sentence shows on
                     screen while you record. All-ASCII, so classic Shift-JIS
                     OREMO handles it fine; recstar uses the same idea.

arctic_txt.done.data Festival transcript: ( arctic_a0001 "sentence" ) per line.
                     What the Multisyn build consumes as label text. Keep the
                     IDs -- wav filenames must match these.

WORKFLOW
--------
1. Record: load arctic_reclist.txt in OREMO/recstar, drop OREMO-comment.txt in
   the output folder, read each sentence once per take. One utterance per wav;
   keep the ID as the filename (arctic_a0001.wav ...).
2. Audio: quiet room, fixed mic/distance/pop filter, 48 kHz / 16-bit mono WAV.
   Neutral, declarative, CONSISTENT pitch/energy/pace across the whole set --
   drift is what makes unit-selection joins audible. Watch clipping, breaths,
   lip-smacks; leave a little head/tail silence.
3. Build: give wav/ + arctic_txt.done.data to the Festival Multisyn build
   (multisyn_build). It runs HTK forced alignment to label phones itself -- you
   do NOT segment the audio by hand.

SUBSET (smaller voice)
----------------------
The "a" set (arctic_a0001..a0593) is a coverage-balanced standalone list.
Record just those for a lighter voice, add "b" later:
    grep '^arctic_a' arctic_reclist.txt      > reclist_a.txt
    grep '^arctic_a' OREMO-comment.txt        > OREMO-comment_a.txt
    grep 'arctic_a[0-9]' arctic_txt.done.data > a_txt.done.data

NEXT (Asaxi)
------------
Same three files, generated from Asaxi text via greedy diphone-coverage
selection, and reclist/comment must be UTF-8 via recstar (classic OREMO's
Shift-JIS cannot encode ng-hook, r-acute, a-breve, u-ring, etc.).
