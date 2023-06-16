from PyQt5.QtWidgets import QMainWindow, QLabel, QMessageBox
import tensorflow as tf


class HelpWindow(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent)

    def show_input_help(self):
        help_text = "Die Größe der Eingabe (Input) definiert die Dimension der Eingabevektoren. Sie sollte der Größe der Merkmalsvektoren entsprechen, die du in " \
                    "deinem Textübersetzungsszenario verwendest.\n\n" \
                    "Beispiel: Wenn du Sätze mit einer Länge von 10 Wörtern übersetzt, gibst du hier 10 ein."
        QMessageBox.information(self, "Hilfe - Input Size", help_text)

    def show_hidden_help(self):
        help_text = "Die versteckte Größe (Hidden Size) definiert die Anzahl der Neuronen im LSTM-Layer. Sie beeinflusst die Kapazität und Lernfähigkeit des Modells. " \
                    "Größere versteckte Größen ermöglichen es dem Modell, komplexere Muster zu erlernen, erfordern jedoch mehr Rechenleistung.\n\n" \
                    "Beispiel: Eine versteckte Größe von 128 bedeutet, dass das LSTM-Layer 128 Neuronen enthält."
        QMessageBox.information(self, "Hilfe - Hidden Size", help_text)

    def show_vocab_help(self):
        help_text = "Die Größe des Vokabulars (Vocabulary Size) definiert die Anzahl der eindeutigen Wörter oder Zeichen in deinem Textkorpus. " \
                    "Es wird verwendet, um die Dimension des Embedding-Layers und die Anzahl der Ausgabeneuronen im finalen Dense-Layer festzulegen.\n\n" \
                    "Beispiel: Wenn dein Textkorpus 10.000 verschiedene Wörter enthält, gibst du hier 10000 ein."
        QMessageBox.information(self, "Hilfe - Vocabulary Size", help_text)

    def show_layers_help(self):
        help_text = "Die Anzahl der LSTM-Schichten definiert, wie viele LSTM-Schichten in das Modell eingefügt werden sollen. " \
                    "Mehr Schichten können dazu beitragen, komplexere Zusammenhänge zu erfassen, erfordern jedoch mehr Rechenleistung und längere Trainingszeiten.\n\n" \
                    "Beispiel: Bei einer Textübersetzung können 2-4 LSTM-Schichten ausreichend sein."
        QMessageBox.information(self, "Hilfe - Anzahl der LSTM-Schichten", help_text)

    def show_dropout_help(self):
        help_text = "Die Dropout-Rate definiert den Prozentsatz der Neuronen, die während des Trainings deaktiviert werden, um Overfitting zu reduzieren. " \
                    "Ein niedrigerer Dropout-Wert bedeutet weniger Deaktivierung und kann zu einer besseren Anpassung an die Trainingsdaten führen, " \
                    "aber auch zu einem erhöhten Risiko von Overfitting.\n\n" \
                    "Beispiel: Eine Dropout-Rate von 0.2 bedeutet, dass 20% der Neuronen während des Trainings deaktiviert werden."
        QMessageBox.information(self, "Hilfe - Dropout-Rate", help_text)

    def show_learning_rate_help(self):
        help_text = "Die Lernrate (Learning Rate) bestimmt die Schrittgröße, mit der das Modell die Gewichte anpasst. " \
                    "Eine niedrigere Lernrate kann zu einer genaueren Anpassung führen, erfordert jedoch möglicherweise mehr Trainingszeit, " \
                    "während eine höhere Lernrate das Modell schneller konvergieren lassen kann, aber möglicherweise zu einer schlechteren Anpassung führt.\n\n" \
                    "Beispiel: Eine Lernrate von 0.001 ist eine häufig verwendete Startpunkt für viele Modelle."
        QMessageBox.information(self, "Hilfe - Lernrate", help_text)

    def show_batch_help(self):
        help_text = "Die Batch-Größe (Batch Size) definiert die Anzahl der Trainingsbeispiele, die in einem Schritt verarbeitet werden. " \
                    "Eine größere Batch-Größe kann zu einer schnelleren Verarbeitung führen, erfordert jedoch mehr Speicherplatz und kann weniger " \
                    "genaue Gewichtsaktualisierungen verursachen.\n\n" \
                    "Beispiel: Eine Batch-Größe von 64 bedeutet, dass 64 Trainingsbeispiele in einem Schritt verarbeitet werden."
        QMessageBox.information(self, "Hilfe - Batch Size", help_text)

    def show_epochs_help(self):
        help_text = "Die Anzahl der Epochen definiert, wie oft das Modell über den gesamten Trainingsdatensatz hinweg trainiert wird. " \
                    "Eine größere Anzahl von Epochen kann zu einer besseren Anpassung führen, aber auch zu einer längeren Trainingszeit und einem " \
                    "höheren Risiko von Overfitting.\n\n" \
                    "Beispiel: Eine Anzahl von 10 Epochen bedeutet, dass das Modell 10-mal über den gesamten Trainingsdatensatz trainiert wird."
        QMessageBox.information(self, "Hilfe - Anzahl der Epochen", help_text)

    def show_optimizer_help(self, optimizer):
        if optimizer == "SGD":
            help_text = "Der SGD-Optimizer (Stochastic Gradient Descent) ist ein Optimierungsalgorithmus, der basierend " \
                        "auf dem Gradienten der Verlustfunktion und einer Lernrate die Gewichte anpasst. SGD eignet sich " \
                        "gut für einfache Modelle und große Datensätze. Bei Übersetzungsmodellen kann SGD beispielsweise " \
                        "verwendet werden, um ein einfaches neuronales Netzwerk für die maschinelle Übersetzung zu trainieren. " \
                        "Das Modell kann dabei Texte in einer Sprache entgegennehmen und sie in eine andere Sprache übersetzen."

        elif optimizer == "Adam":
            help_text = "Der Adam-Optimizer (Adaptive Moment Estimation) ist ein Optimierungsalgorithmus, der den Gradienten " \
                        "der Verlustfunktion basierend auf vergangenen Gradientenmomenten anpasst. Adam eignet sich gut " \
                        "für tiefe neuronale Netze und komplexe Modelle. In Übersetzungsmodellen kann Adam beispielsweise " \
                        "verwendet werden, um ein komplexes Transformer-Modell für die maschinelle Übersetzung zu trainieren. " \
                        "Das Modell kann dabei die Aufgabe haben, Texte in einer Sprache in Texte einer anderen Sprache zu " \
                        "übersetzen und dabei komplexe sprachliche Strukturen zu berücksichtigen."

        elif optimizer == "RMSprop":
            help_text = "Der RMSprop-Optimizer (Root Mean Square Propagation) ist ein Optimierungsalgorithmus, der die " \
                        "Lernrate basierend auf einem gleitenden Durchschnitt der Quadratsumme der vorherigen Gradienten " \
                        "anpasst. RMSprop eignet sich gut für nicht konvexe Optimierungsprobleme. In Übersetzungsmodellen " \
                        "kann RMSprop beispielsweise verwendet werden, um ein neuronales Netzwerk für die maschinelle " \
                        "Übersetzung zu trainieren, das auf einer Rekurrenten-Neuralen-Netzwerk-Architektur basiert. " \
                        "Das Modell kann dabei Sequenzen von Wörtern in einer Sprache in Sequenzen von Wörtern in einer " \
                        "anderen Sprache übersetzen."

        elif optimizer == "Adagrad":
            help_text = "Der Adagrad-Optimizer (Adaptive Gradient Algorithm) passt die Lernrate für jedes Gewicht basierend " \
                        "auf der Häufigkeit der Aktualisierungen an. Adagrad eignet sich gut für sparse Daten oder Probleme " \
                        "mit unterschiedlicher Varianz in den Features. In Übersetzungsmodellen kann Adagrad beispielsweise " \
                        "verwendet werden, um ein Modell für die maschinelle Übersetzung von Texten mit seltenen Wörtern " \
                        "zu trainieren. Das Modell kann dabei selten vorkommende Wörter in den Trainingsdaten effektiv " \
                        "berücksichtigen und bei der Übersetzung seltener Wörter genauere Ergebnisse erzielen."

        elif optimizer == "Adadelta":
            help_text = "Der Adadelta-Optimizer ist ein Optimierungsalgorithmus, der die Lernrate basierend auf einer " \
                        "Schätzung des Gradientenupdates und der vergangenen Gradientenmomenten anpasst. Adadelta ist " \
                        "ein verbesserter Optimierer des Adagrad-Algorithmus. In Übersetzungsmodellen kann Adadelta beispielsweise " \
                        "verwendet werden, um ein Modell für die maschinelle Übersetzung zu trainieren, das mit " \
                        "unregelmäßigen Gradientenupdates und Lernratenanpassungen gut umgehen kann. Das Modell kann " \
                        "dabei verschiedene Schritte in der Übersetzungsaufgabe ausführen, wie z. B. Kodierung, Decodierung " \
                        "und Aufmerksamkeitsmechanismen."

        elif optimizer == "Adamax":
            help_text = "Der Adamax-Optimizer ist eine Variante des Adam-Optimierers, die eine unendliche Norm (Max-Norm) " \
                        "für die Gewichte verwendet. Adamax eignet sich gut für Modelle mit unendlicher Gewichtsnorm. " \
                        "In Übersetzungsmodellen kann Adamax beispielsweise verwendet werden, um ein Modell für die " \
                        "maschinelle Übersetzung mit einer speziellen Aufmerksamkeitsmechanismus-Architektur zu trainieren. " \
                        "Das Modell kann dabei verschiedene Aufmerksamkeitsgewichte für die Wortübersetzung berechnen " \
                        "und so die Leistung bei der Übersetzung verbessern."

        elif optimizer == "Nadam":
            help_text = "Der Nadam-Optimizer kombiniert die Vorteile von Adam und Nesterov Momentum. Er passt den " \
                        "Gradienten basierend auf dem Adam-Update und einem Nesterov Momentum-Update an. In Übersetzungsmodellen " \
                        "kann Nadam beispielsweise verwendet werden, um ein Modell für die maschinelle Übersetzung mit " \
                        "rekurrenten neuronalen Netzwerken (RNNs) und einem verbesserten Nesterov Momentum zu trainieren. " \
                        "Das Modell kann dabei Wörter in einer Eingabesequenz schrittweise verarbeiten und die Übersetzung " \
                        "schrittweise generieren."

        elif optimizer == "Ftrl":
            help_text = "Der Ftrl-Optimizer (Follow-the-Regularized-Leader) ist ein Optimierungsalgorithmus, der " \
                        "speziell für lineare Modelle mit großen, sparse Daten entwickelt wurde. Er verwendet eine " \
                        "kombinierte Regularisierung und Lernratenplanung. In Übersetzungsmodellen sind Ftrl-Optimierer " \
                        "möglicherweise weniger relevant, da sie hauptsächlich auf lineare Modelle abzielen und " \
                        "Übersetzungsmodelle in der Regel komplexere Architekturen verwenden."

        elif optimizer == "Adafactor":
            help_text = "Der Adafactor-Optimizer ist ein Optimierungsalgorithmus, der eine adaptive Lernrate verwendet " \
                        "und die Vorteile von Adam und Adagrad kombiniert. Er eignet sich gut für Modelle mit vielen " \
                        "Parametern. In Übersetzungsmodellen kann Adafactor beispielsweise verwendet werden, um ein " \
                        "Transformer-Modell für die maschinelle Übersetzung mit adaptiver Lernrate zu trainieren. " \
                        "Das Modell kann dabei auf großen Datensätzen mit vielen Parametern arbeiten und die Lernrate " \
                        "an die spezifischen Anforderungen der Übersetzungs-aufgabe anpassen."

        elif optimizer == "AdamW":
            help_text = "Der AdamW-Optimizer ist eine Variante des Adam-Optimierers, der zusätzlich zur Gewichtsaktualisierung " \
                        "eine L2-Regularisierung durchführt. AdamW eignet sich gut für Modelle mit feiner Abstimmung und " \
                        "Transfer Learning. In Übersetzungsmodellen kann AdamW beispielsweise verwendet werden, um ein " \
                        "vortrainiertes Modell für die maschinelle Übersetzung zu feinabstimmen. Das Modell kann dabei " \
                        "auf einem großen, allgemeinen Sprachdatensatz vortrainiert sein und mit AdamW an die spezifische " \
                        "Übersetzungsaufgabe angepasst werden."

        else:
            help_text = "Es tut mir leid, aber es steht keine Beschreibung für den ausgewählten Optimierer zur Verfügung."

        QMessageBox.information(self, f"Hilfe - {optimizer} Optimizer", help_text)

    def show_loss_help(self, loss):
        if loss == "mean_squared_error":
            help_text = "Mean Squared Error (MSE) ist eine gängige Verlustfunktion für Regressionen. Sie berechnet den " \
                        "durchschnittlichen quadratischen Fehler zwischen den vorhergesagten und den tatsächlichen Werten. " \
                        "MSE eignet sich gut für Regressionsszenarien."

        elif loss == "mean_absolute_error":
            help_text = "Mean Absolute Error (MAE) ist eine Verlustfunktion für Regressionen. Sie berechnet den " \
                        "durchschnittlichen absoluten Fehler zwischen den vorhergesagten und den tatsächlichen Werten. " \
                        "MAE eignet sich gut für Regressionsszenarien."

        elif loss == "mean_absolute_percentage_error":
            help_text = "Mean Absolute Percentage Error (MAPE) ist eine Verlustfunktion für Regressionen, die den " \
                        "prozentualen Fehler zwischen den vorhergesagten und den tatsächlichen Werten berechnet. " \
                        "MAPE eignet sich gut für Regressionsszenarien."

        elif loss == "mean_squared_logarithmic_error":
            help_text = "Mean Squared Logarithmic Error (MSLE) ist eine Verlustfunktion für Regressionen, die den " \
                        "logarithmierten quadratischen Fehler zwischen den vorhergesagten und den tatsächlichen Werten " \
                        "berechnet. MSLE eignet sich gut für Regressionsszenarien."

        elif loss == "binary_crossentropy":
            help_text = "Binary Crossentropy ist eine Verlustfunktion für binäre Klassifikationsprobleme. Sie berechnet " \
                        "den Kreuzentropie-Fehler zwischen den vorhergesagten und den tatsächlichen Wahrscheinlichkeiten " \
                        "für die beiden Klassen. Binary Crossentropy eignet sich gut für binäre Klassifikationsszenarien."

        elif loss == "categorical_crossentropy":
            help_text = "Categorical Crossentropy ist eine Verlustfunktion für Klassifikationsprobleme mit mehreren " \
                        "Klassen. Sie berechnet den Kreuzentropie-Fehler zwischen den vorhergesagten und den tatsächlichen " \
                        "Wahrscheinlichkeitsverteilungen. Categorical Crossentropy eignet sich gut für Klassifikationsszenarien."

        elif loss == "sparse_categorical_crossentropy":
            help_text = "Sparse Categorical Crossentropy ist ähnlich wie Categorical Crossentropy, wird aber verwendet, " \
                        "wenn die Klassenlabels als ganze Zahlen dargestellt werden, anstatt als One-Hot-Vektoren. Diese " \
                        "Verlustfunktion eignet sich gut für Übersetzungsmodelle."

        elif loss == "kullback_leibler_divergence":
            help_text = "Kullback-Leibler Divergence (KL-Divergenz) ist eine Verlustfunktion, die den Unterschied zwischen " \
                        "zwei Wahrscheinlichkeitsverteilungen misst. KL-Divergenz eignet sich gut für Szenarien, in denen " \
                        "die Divergenz zwischen Wahrscheinlichkeitsverteilungen wichtig ist."

        elif loss == "poisson":
            help_text = "Poisson ist eine Verlustfunktion für Regressionen, die auf der Poisson-Verteilung basiert. Sie " \
                        "eignet sich gut für Szenarien, in denen die vorhergesagten Werte diskrete positive Zahlen " \
                        "darstellen."

        elif loss == "cosine_similarity":
            help_text = "Cosine Similarity ist eine Verlustfunktion, die den kosinusbasierten Ähnlichkeitswert zwischen " \
                        "zwei Vektoren berechnet. Sie eignet sich gut für Szenarien, in denen die Ähnlichkeit zwischen " \
                        "Vektoren wichtig ist."

        # Füge hier weitere Loss-Funktionen hinzu

        else:
            help_text = "Es tut mir leid, aber es steht keine Beschreibung für den ausgewählten Loss zur Verfügung."

        QMessageBox.information(self, f"Hilfe - {loss} Loss", help_text)

