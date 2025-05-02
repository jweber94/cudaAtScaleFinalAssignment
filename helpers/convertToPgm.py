#!/usr/bin/python3
import os
import sys
from PIL import Image
import numpy as np

def tiff_ordner_zu_pgm(eingabe_ordner):
    """
    Konvertiert alle TIFF-Dateien in einem Ordner in PGM-Dateien.

    Args:
        eingabe_ordner (str): Der Pfad zum Eingabe-Ordner mit TIFF-Dateien.
    """
    if not os.path.isdir(eingabe_ordner):
        print(f"Fehler: Der angegebene Pfad '{eingabe_ordner}' ist kein gültiger Ordner.")
        return

    tiff_dateien = [f for f in os.listdir(eingabe_ordner) if f.lower().endswith(('.tiff', '.tif'))]

    if not tiff_dateien:
        print(f"Keine TIFF-Dateien im Ordner '{eingabe_ordner}' gefunden.")
        return

    print(f"Starte Konvertierung von {len(tiff_dateien)} TIFF-Dateien im Ordner '{eingabe_ordner}'.")

    for tiff_datei_name in tiff_dateien:
        tiff_pfad = os.path.join(eingabe_ordner, tiff_datei_name)
        basis_name, _ = os.path.splitext(tiff_datei_name)
        pgm_datei_name = basis_name + ".pgm"
        pgm_pfad = os.path.join(eingabe_ordner, pgm_datei_name)

        try:
            # TIFF-Bild öffnen
            img = Image.open(tiff_pfad)

            # In Graustufen konvertieren, falls es kein Graustufenbild ist
            if img.mode != 'L':
                img = img.convert('L')

            # Bilddaten als NumPy-Array erhalten
            bild_array = np.array(img)
            hoehe, breite = bild_array.shape

            # PGM-Datei im Binärformat schreiben
            with open(pgm_pfad, 'wb') as pgm_datei:
                # PGM-Header schreiben
                pgm_datei.write(b'P5\n')  # Magische Zahl für binäres PGM
                pgm_datei.write(f'{breite} {hoehe}\n'.encode('ascii'))
                pgm_datei.write(b'255\n')  # Maximaler Grauwert

                # Bilddaten schreiben
                pgm_datei.write(bild_array.tobytes())

            print(f"Konvertiert: '{tiff_datei_name}' -> '{pgm_datei_name}'")

        except FileNotFoundError:
            print(f"Fehler: Die Datei '{tiff_pfad}' wurde nicht gefunden (sollte nicht passieren).")
        except Exception as e:
            print(f"Fehler bei der Konvertierung von '{tiff_datei_name}': {e}")

    print("Konvertierung aller TIFF-Dateien abgeschlossen.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Verwendung: python dein_skriptname.py <pfad_zum_tiff_ordner>")
        sys.exit(1)

    eingabe_ordner_pfad = sys.argv[1]
    tiff_ordner_zu_pgm(eingabe_ordner_pfad)