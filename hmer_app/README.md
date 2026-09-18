# Handwriting → LaTeX

Local web app. Drop in an image, get LaTeX.

    ./run.sh                 # then open http://localhost:8000

Standard library only — nothing to install beyond what the project already
uses. The model loads once at startup.

## Options

    ./run.sh --port 8080
    ./run.sh --checkpoint ../best_model_96px.pt
    ./run.sh --device cpu

## Input modes

Pick the one matching where the image came from; the wrong one changes the
answer completely.

| mode | for | on the test equation |
|---|---|---|
| Drawing / screenshot | drawing apps, screenshots, thick strokes | `x^{2}+y^{2}=4` ✅ |
| Photo of paper | real photographs | `\begin{matrix}20\\ 2\end{matrix}3` ❌ |
| Already a render | images already in the training convention | `g^{2}+\frac{g^{2}}{d}=4` ❌ |

"Drawing / screenshot" re-renders strokes at the uniform 2.5px width the model
trained on (see `inkml_normalize.py`). "Photo of paper" runs perspective
correction and page detection, which needs an actual page to find.

The app always shows the preprocessed image alongside the LaTeX. When a
prediction looks wrong, that image usually explains it.
