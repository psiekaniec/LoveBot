# LoveBot
LoveBot to cichy bohater social media - czynnie staje po stronie hejtowanego dziecka, pisząc coś miłego. Ma unikalne, własne słowniki!

Część praktyczna składa się z dwóch części: własnego czatu i bota. 
Pierwszy z nich został stworzony w języku Python wraz z odpowiednimi bibliotekami (obecnie tylko wersja desktop). Bot opiera się na głębokiej sieci neuronowej z 2 warstwami połączonymi wg schematu "każdy z każdym", jedną warstwą z funkcją aktywacji softmax i jedną warstwą regresji (1. wariant) oraz modelu Sequential wraz z dwiema warstwami typu Dense i warstwą z funkcją aktywacji softmax (2. wariant). Do budowy tych modeli użyto bilbiotek (odpowiednio) TfLearn w wersji 0.5.0 oraz Tensorflow w wersji 2.5.0 i Keras w wersji 2.4.3 wraz z innymi pomocniczymi pakietami.

Cały projekt jest efektem współpracy 2 osób w ramach hackathonu #hack4lem organizowanego przez Microsoft, PKO BP i Instytut Polska Przyszłości (opis na: https://challengerocket.com/hack4lem/works/lovebot-02450e#go-pagecontent).
