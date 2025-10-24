🔍 Was geändert & warum
Kategorie	Änderung	Grund
📊 preprocessing	scaler: minmax, scale_per_symbol: true	verhindert Gradient-Explosion und Bias durch große Preisunterschiede
⚖️ class_weight: balanced	aktiviert	gleicht Down/Up-Verhältnis automatisch aus
🔁 return_sequences	[true, true, false]	nötig, um LSTM-Schichten korrekt zu verbinden
🧩 Dropout-Werte erhöht	CNN: 0.4 / LSTM: 0.4,0.3,0.2	senkt Overfitting auf Rauschen
⏱️ interval: 5m	weniger Marktrauschen als 1m	
⚔️ confidence_threshold	0.7	nur sichere Vorhersagen für Trades
🧮 shuffle: false	Zeitreihen dürfen nicht zufällig gemischt werden	
🧾 paper simulation aktiviert	um Performance ohne Risiko zu messen