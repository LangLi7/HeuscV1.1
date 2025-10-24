import freecurrencyapi
import os
from dotenv import load_dotenv

# Lade .env aus übergeordnetem Verzeichnis (eine Ebene höher als das Modul)
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)


class Investment:
    def __init__(self, kaufpreis, verkaufspreis, investiert_betrag, gebuehren=0):
        """
        Initialisiert die Investment-Daten.

        :param kaufpreis: Preis pro Einheit beim Kauf (CHF pro Einheit)
        :param verkaufspreis: Preis pro Einheit beim Verkauf (CHF pro Einheit)
        :param investiert_betrag: Investierter Gesamtbetrag in CHF
        :param gebuehren: Gebühren für Kauf und Verkauf in CHF (optional)
        """
        self.kaufpreis = kaufpreis
        self.verkaufspreis = verkaufspreis
        self.investiert_betrag = investiert_betrag
        self.gebuehren = gebuehren

    def berechne_gewinn(self):
        """
        Berechnet und gibt den Gewinn in CHF zurück.
        """
        menge = self.investiert_betrag / self.kaufpreis
        bruttowert = menge * self.verkaufspreis
        gewinn = bruttowert - self.investiert_betrag - self.gebuehren
        return gewinn

    def waehrungsumrechnung(self, betrag, von_waehrung, zu_waehrung):
        """
        Rechnet einen Betrag von einer Währung in eine andere um.

        :param betrag: Betrag in der Ausgangswährung
        :param von_waehrung: Ausgangswährung z.B. 'CHF'
        :param zu_waehrung: Zielwährung z.B. 'USD', 'EUR'
        :return: Umgerechneter Betrag in Zielwährung
        """
        api_key = os.getenv('FREECURRENCYAPI_KEY')
        if not api_key:
            raise ValueError("FREECURRENCYAPI_KEY nicht in .env gefunden!")

        try:
            client = freecurrencyapi.Client(api_key)
            result = client.latest(base_currency=von_waehrung)
            
            if 'data' not in result:
                raise ValueError("Ungültige API Antwort")
            
            kurs_dict = result['data']
            if zu_waehrung not in kurs_dict:
                raise ValueError(f"Keine Umrechnung für {zu_waehrung} verfügbar.")
            
            return betrag * kurs_dict[zu_waehrung]
        except Exception as e:
            raise ConnectionError(f"Währungsumrechnung fehlgeschlagen: {e}")


if __name__ == "__main__":
    # Debug: Zeige, wo die .env gesucht wird
    print(f"Suche .env in: {os.path.abspath(env_path)}")
    print(f"API Key geladen: {os.getenv('FREECURRENCYAPI_KEY') is not None}")
    
    kaufpreis = 20000
    verkaufspreis = 30000
    investiert_betrag = 100
    gebuehren = 1
    
    inv = Investment(kaufpreis, verkaufspreis, investiert_betrag, gebuehren)
    print(f"\nkaufpreis: {kaufpreis}")
    print(f"verkaufspreis: {verkaufspreis}")
    print(f"investiert_betrag: {investiert_betrag}")
    print(f"Gewinn: {inv.berechne_gewinn():.2f} CHF")
    
    # Währungsumrechnung testen
    try:
        betrag_usd = inv.waehrungsumrechnung(investiert_betrag, 'CHF', 'USD')
        print(f"\n{investiert_betrag} CHF = {betrag_usd:.2f} USD")
        
        gewinn_eur = inv.waehrungsumrechnung(inv.berechne_gewinn(), 'CHF', 'EUR')
        print(f"Gewinn in EUR: {gewinn_eur:.2f} EUR")
    except Exception as e:
        print(f"Fehler bei Währungsumrechnung: {e}")
