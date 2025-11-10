import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
#from duckduckgo_search import DDGS
import urllib.request
#import duckduckgo_search
import argparse
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity  # Für Matching
import time  # Für genaue Zeitsteuerung, falls benötigt
from tqdm import tqdm  # Für Fortschrittsanzeige beim Verarbeiten
import requests
import threading  # Für asynchrone API-Calls
from scipy.spatial.distance import cdist  # Für vektorisierte Similarity
import time  # Für Pausen bei Rate Limits
from PIL import Image # Image Library zur Benutzung mit Tesseract
import pytesseract # Tesseract Wrapper

# Liste der Top LPGA-Spielerinnen (erweitert basierend auf 2025-Rankings)
PLAYERS = [
"Casandra Alexander",
"Helen Alfredsson",
"Carmen Alonso",
"Ela Anacona",
"Pajaree Anannarukarn",
"April Angurasaranee",
"Tiffany Arafi",
"Amanda Ard",
"Kajsa Arwefjall",
"Aditi Ashok",
"Aloysa Atienza",
"Miriam Ayora",
"Pia Babnik",
"Lianna Bailey",
"Jess Baker",
"Hitaashee Bakshi",
"Susan Bamford",
"Ana Belac",
"Rosie Belsham",
"Kelsey Bennett",
"Laura Beveridge",
"Gudrun Bjorgvinsdottir",
"Carly Booth",
"Celine Borge",
"Annika Borrelli",
"Justice Bosio",
"Amy Boulden",
"Celine Boutier",
"Vanessa Bouvet",
"Lyn Brand",
"Stacy Bregman",
"Becky Brewerton",
"Helen Briem",
"Sofie Bringner",
"Nicole Broch Estrup",
"Samantha Bruce",
"Ashleigh Buhai",
"Kan Bunnabodee",
"Hannah Burke",
"Maxine Burton",
"Sara Byrne",
"Virginia Elena Carta",
"Anne-Lise Caudal",
"Tiffany Chan",
"Trichat Cheenglab",
"Camille Chevalier",
"Pamela Chugg MBE",
"Carlota Ciganda",
"Gemma Clews",
"Holly Clyburn",
"Pasqualle Coffa",
"Selena Costabile",
"Olivia Cowan",
"Gabriella Cowley",
"Diksha Dagar",
"Daniela Darquea",
"Klara Davidson Spilkova",
"Laura Davies",
"Rosie Davies",
"Hayley Davis",
"Ana Dawson",
"Marie Laure De Lorenzi",
"Manon De Roey",
"Perrine Delacour",
"Jane Denman",
"Megan Dennis",
"Teresa Diez Moliner",
"Annabel Dimmock",
"Ginnie Ding",
"Amandeep Drall",
"Gemma Dryburgh",
"Danielle Du Toit",
"Mia Eales-Smith",
"Margaret Egan",
"Karstin Ehrnlund",
"Marina Escobar Domingo",
"Jodi Ewart Shadoff",
"Alessandra Fanali",
"Natasha Fear",
"Fatima Fernandez Cano",
"Blanca Fernandez",
"Michaela Finn",
"Cecilie Finne-Ipsen",
"Moa Folke",
"Dorthea Forbrigd",
"Jane Forrest",
"Michelle Forsland",
"Alexandra Forsterling",
"Anna Foster",
"Moa Fridell",
"Laura Fuenfstueck",
"Reina Fujikawa",
"Annabell Fuller",
"Cara Gainer",
"Sandra Gal",
"Luna Sobron Galmes",
"Nicole Garcia",
"Amelia Garvey",
"Sarah Gee",
"Verena Gimmy",
"Eleanor Givens",
"Laura Gomez Ruiz",
"Alice Gotbring",
"Ellie Gower",
"Linn Grant",
"Emma Grechi",
"Chris Green",
"Hannah Gregg",
"Christine Griffith",
"Natascha Grossschadl-fink",
"Nataliya Guseva",
"Sophie Gustafson",
"Maha Haddioui",
"Georgia Hall",
"Lydia Hall",
"Esme Hamilton",
"Leonie Harm",
"Lynne Harrold",
"Darcey Harry",
"Denise Hastings",
"Sophie Hausmann",
"Caroline Hedwall",
"Kylie Henry",
"Esther Henseleit",
"Celine Herbin",
"Maria Hernandez",
"Maria Herraez Galvez",
"Alice Hewson",
"Whitney Hillier",
"Maddison Hinson-Tolchard",
"Nikki Hofstede",
"Katie Hollern",
"Lauren Holmey",
"Mayka Hoogeboom",
"Samantha Hortin-Giles",
"Anna Huang",
"Beverly Huke",
"Charley Hull",
"Ellen Hume",
"Lily May Humphreys",
"Natacha Host Husted",
"Nuria Iturrioz",
"Belinda Ji",
"Pearl Jin",
"Linnea Johansson",
"Trish Johnson",
"Stephanie Alderlieste",
"Vani Kapoor",
"Hannah Karg",
"Carolin Kauffmann",
"Wenyung Keh",
"Sarah Kemp",
"In-Kyung Kim",
"Sara Kjellker",
"Ariane Klotz",
"Vanessa Knecht",
"Momoka Kobori",
"Tiia Koivisto",
"Noora Komulainen",
"Sara Kouskova",
"Tereza Kozeluhova",
"Aline Krauter",
"Helen Tamy Kreuzer",
"Ragga Kristinsdottir",
"Karina Kukkonen",
"Stephanie Kyriacou",
"Charlotte Laffar",
"Agathe Laisne",
"Ines Laklalech",
"Christine Langford",
"Amaia Latorre",
"Lois Lau",
"Bronte Law",
"Alison Lee",
"Harang Lee",
"Camilla Lennarth",
"Amalie Leth-Nissen",
"Charlotte Liautier",
"Xiyu Lin",
"Pernilla Lindberg",
"Fernanda Lira",
"Jenny Lee Lucas",
"Karoline Lund",
"Kelsey Macdonald",
"Meghan MacLaren",
"Polly Mack",
"Patricie Mackova",
"Nanna Koerstz Madsen",
"Anna Magnusson",
"Leona Maguire",
"Lucie Malchirand",
"Paz Marfa Sans",
"Marta Martin",
"Thalia Martin",
"Vanessa Marvin",
"Caroline Masson",
"Catriona Matthew",
"Tina Mazarino",
"Lorna Mcclymont",
"Hannah McCook",
"Romy Meekers",
"Olivia Mehaffey",
"Tereza Melecka",
"Jana Melichova",
"Kim Metraux",
"Morgane Metraux",
"Anais Meyssonnier",
"Kaiyuree Moodley",
"Susan Moon",
"Elena Moosmann",
"Anne-Charlotte Mora",
"Clara Moyano Reigosa",
"Katharina Muehlbauer",
"Azahara Munoz",
"Nastasia Nadaud",
"Kristyna Napoleaova",
"Brianna Navarrosa",
"Liselotte Neumann",
"Alison Nicholas",
"Sofie Kibsgaard Nielsen",
"Alessia Nobilio",
"Chiara Noja",
"Anna Nordqvist",
"Elina Nummenpaa",
"Sanna Nuutinen",
"Georgia Iziemgbe Oboh",
"Fie Olsen",
"Lee-Anne Pace",
"Alexa Pano",
"Cathy Panton-Lewis",
"Do Yeon Park",
"Florentyna Parker",
"Maria Parra",
"Emily Kristine Pedersen",
"Ana Pelaez Trivino",
"Emily Penttila",
"Marta Perez",
"Emie Peronnin",
"Suzann Pettersen",
"Lisa Pettersson",
"Katja Pogacar",
"Avani Prashanth",
"Mireia Prat",
"Leticia Ras-Anderica",
"Mel Reid",
"Mimi Rhodes",
"Pauline Roussin-Bouchard",
"Kirsten Rudgeley",
"Madelene Sagstrom",
"Elina Saksa",
"Chloe Salort",
"Tvesa Malik",
"Supamas Sangchan",
"Marta Sanz Barrio",
"Celina Sattelkau",
"Vivien Saunders OBE",
"Agathe Sauzon",
"Priscilla Schmid",
"Olivia Schmidt",
"Patricia Isabel Schmidt",
"Sarah Schober",
"Hannah Screen",
"Canice Screene",
"Ashely Shim",
"Magdalena Simmermacher",
"Sneha Singh",
"Marianne Skarpnord",
"Zoe Slaughter",
"Laura Sluman",
"Billie-Jo Smith",
"Jo Jeffreys",
"Smilla Tarning Soenderby",
"Kimberley Sommer",
"Annika Sorenstam",
"Emma Spitz",
"Louise Stahle",
"Maja Stark",
"Madelene Stavnar",
"Linnea Strom",
"Ellinor Sudow",
"Alexandra Swayne",
"Chiara Tamburlini",
"Shannon Tan",
"Amy Taylor",
"Lauren Taylor",
"Tia Teiniketo",
"Jeeno Thitikul",
"Puk Lyng Thomsen",
"Michele Thomson",
"Maria Fernanda Torres",
"Teresa Toscano",
"Neha Tripathi",
"Jane Turner",
"Ayako Uehara",
"Mariajo Uribe",
"Pranavi Urs",
"Vidhatri Urs",
"Aunchisa Utama",
"Albane Valenzuela",
"Anne Van Dam",
"Corinne Viden",
"Sophie Walker",
"Mickey Walker Obe",
"Amy Walsh",
"Lauren Walsh",
"Linda Wang",
"Linda Wessberg",
"Ursula Wikstrom",
"Chloe Williams",
"Annabel Wilson",
"Christine Wolf",
"Johanna Wrigley",
"Anne Wynn",
"Ayeon Yang",
"Liz Young",
"Dorota Zalewska",
"Anna Zanusso"
] 


# ---------- KONSTANTEN  ----------
API_URL = "http://localhost:5000/update" # API-URL für den Webserver (grafik.py)
# Ordner für Dataset und Embeddings
DATASET_DIR = "lpga_dataset"
EMBEDDINGS_FILE = "embeddings.pkl"

# Übergabe des Pfads zur tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# ---------- INITIALISIERUNG DER MODELLE  ----------
# InsightFace-Modell initialisieren (ArcFace für Recognition, RetinaFace für Detection)
app = FaceAnalysis(name='buffalo_l')  # 'buffalo_l' ist ein starkes Modell für Accuracy
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 für GPU; -1 für CPU


# ---------- DATENSATZ GENERIERUNG ÜBER WEBSUCHE  ----------
def build_dataset(max_images_per_player=200):
    """Sammle und croppe Bilder pro Spielerin via DuckDuckGo."""
    # Lade Fortschritt (letzter erfolgreicher Player-Index)
    start_index = 0
    if os.path.exists('progress.json'):
        with open('progress.json', 'r') as f:
            progress = json.load(f)
            start_index = progress.get('last_player_index', 0)
        print(f"Fortfahren ab Player-Index {start_index}: {PLAYERS[start_index] if start_index < len(PLAYERS) else 'Ende'}")
    
    for idx in range(start_index, len(PLAYERS)):
        player = PLAYERS[idx]
        player_dir = os.path.join(DATASET_DIR, player.replace(" ", "_"))
        os.makedirs(player_dir, exist_ok=True)
        
        # Suche nach Bildern mit Retry bei Rate Limit
        image_urls = []
        retry_count = 0
        max_retries = 5  # Max Versuche pro Player
        while retry_count < max_retries:
            try:
                with DDGS() as ddgs:
                    results = ddgs.images(keywords=f"{player} LPGA golfer portrait face", max_results=max_images_per_player)
                    image_urls = [result['image'] for result in results]
                break  # Erfolg: Aus Schleife ausbrechen
            except duckduckgo_search.exceptions.RatelimitException as e:
                print(f"Rate Limit für {player}: {e}. Warte 60 Sekunden und versuche erneut (Versuch {retry_count + 1}/{max_retries}).")
                time.sleep(60)  # Pause bei Rate Limit
                retry_count += 1
            except Exception as e:
                print(f"Fehler bei Suche für {player}: {e}. Speichere Fortschritt und starte neu.")
                # Speichere aktuellen Index (für Restart beim aktuellen Player)
                with open('progress.json', 'w') as f:
                    json.dump({'last_player_index': idx}, f)
                return  # Beende Funktion, User startet neu
        
        if not image_urls and retry_count >= max_retries:
            print(f"Maximale Retries für {player} erreicht. Überspringe.")
            continue
        
        # Herunterladen und Cropping mit InsightFace
        for i, url in enumerate(image_urls):
            try:
                temp_path = f"{player_dir}/temp_{i}.jpg"
                # Download mit Timeout
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    with open(temp_path, 'wb') as f:
                        f.write(response.read())
                
                img = cv2.imread(temp_path)
                if img is None: continue
                
                # Prüfe Bildbreite
                if img.shape[1] > 1200:  # Bildbreite > 1200px: Ganzes Bild speichern
                    save_path = f"{player_dir}/full_{i}.jpg"
                    cv2.imwrite(save_path, img)
                    print(f"Ganzes Bild gespeichert: {player} Bild {i} (Breite: {img.shape[1]}px)")
                else:  # Andernfalls: Crop speichern, wenn Gesicht gefunden
                    faces = app.get(img)
                    if len(faces) > 0:
                        best_face = faces[0]  # Beste Detection
                        bbox = best_face.bbox.astype(int)
                        crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        # Nur speichern, wenn Crop größer als 500px (Höhe und Breite)
                        if crop.shape[0] > 500 and crop.shape[1] > 500:
                            cv2.imwrite(f"{player_dir}/face_{i}.jpg", crop)
                            print(f"Gespeichert: {player} Bild {i} (Größe: {crop.shape[1]}x{crop.shape[0]})")
                        else:
                            print(f"Übersprungen: {player} Bild {i} (Größe zu klein: {crop.shape[1]}x{crop.shape[0]})")
                    else:
                        print(f"Kein Gesicht gefunden: {player} Bild {i}")
                
                os.remove(temp_path)
            except urllib.error.URLError as e:
                if 'timeout' in str(e).lower():
                    print(f"Timeout bei {url} für {player}. Speichere Fortschritt und starte neu.")
                    # Speichere aktuellen Index (für Restart beim aktuellen Player)
                    with open('progress.json', 'w') as f:
                        json.dump({'last_player_index': idx}, f)
                    # Neu starten: In realem Einsatz könnte man hier sys.exit(0) verwenden oder das Skript manuell neu starten
                    return  # Für jetzt: Beende Funktion, User startet neu
                else:
                    print(f"Fehler bei {url}: {e}")
            except Exception as e:
                print(f"Fehler bei {url}: {e}")
        
        # Nach erfolgreichem Player: Update Fortschritt zum nächsten
        with open('progress.json', 'w') as f:
            json.dump({'last_player_index': idx + 1}, f)
        print(f"Player {player} abgeschlossen. Fortschritt gespeichert.")

def generate_embeddings():
    """Generiere Embeddings für das Dataset und speichere sie."""
    embeddings = {}
    for player in PLAYERS:
        player_dir = os.path.join(DATASET_DIR, player.replace(" ", "_"))
        if not os.path.exists(player_dir) or not os.listdir(player_dir):
            print(f"Kein Ordner oder Bilder für {player} gefunden. Führe --build_dataset aus.")
            continue
        
        player_embeddings = []
        for img_file in os.listdir(player_dir):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(player_dir, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Bild konnte nicht geladen werden: {img_path} für {player}. Überspringe.")
                    continue
                
                faces = app.get(img)
                if len(faces) > 0:
                    player_embeddings.append(faces[0].embedding)
                    print(f"Embedding generiert für {player}: {img_file}")
                else:
                    print(f"Kein Gesicht in {img_file} für {player} gefunden.")
        
        if player_embeddings:
            # Durchschnittliches Embedding pro Spielerin für Robustheit
            embeddings[player] = np.mean(player_embeddings, axis=0)
            print(f"Durchschnittliches Embedding für {player} gespeichert ({len(player_embeddings)} Bilder).")
        else:
            print(f"Keine Embeddings für {player} generiert – überprüfe Bilder.")
    
    if not embeddings:
        print("Keine Embeddings generiert – Dataset ist leer oder fehlerhaft.")
        return
    
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings gespeichert: {len(embeddings)} Spielerinnen.")

# ---------- HILFSFUNKTIONEN  ----------
def process_and_save_video(video_path, output_path, threshold=0.6):
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings_db = pickle.load(f)
    except FileNotFoundError:
        print("Embeddings-Datei nicht gefunden. Führe --generate_embeddings zuerst aus.")
        return
    
    if not embeddings_db:
        print("Keine Embeddings in der Datenbank. Generiere sie zuerst mit --generate_embeddings.")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Video konnte nicht geöffnet werden: {video_path}")
        return
    
    # Video-Parameter holen
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # VideoWriter für Ausgabe
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4-Codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Ausgabevideo konnte nicht erstellt werden: {output_path}")
        cap.release()
        return
    
    print(f"Verarbeite Video: {total_frames} Frames...")
    
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detection und Recognition
        faces = app.get(frame)
        for face in faces:
            embedding = face.embedding.reshape(1, -1)
            similarities = {player: cosine_similarity(embedding, emb.reshape(1, -1))[0][0] for player, emb in embeddings_db.items()}
            
            if not similarities:
                continue  # Keine DB – überspringen
            
            best_match = max(similarities, key=similarities.get)
            score = similarities[best_match]
            
            if score > threshold:
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{best_match} ({score:.2f})", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Unerkannte Gesichter werden nicht markiert
        
        # Markierten Frame schreiben
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Markiertes Video gespeichert: {output_path}")


# ---------- INFERENCE FUNKTIONEN  ----------
def real_time_api_update(video_source=0, threshold=0.6):
    """Echtzeit-Recognition mit optimierter Verarbeitung und API-Update bei Namensänderung."""
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings_db = pickle.load(f)
    except FileNotFoundError:
        print("Embeddings-Datei nicht gefunden. Führe --generate_embeddings zuerst aus.")
        return
    
    if not embeddings_db:
        print("Keine Embeddings in der Datenbank. Generiere sie zuerst mit --generate_embeddings.")
        return
    
    # Embeddings-DB vektorisiert für Batch-Vergleiche
    player_names = list(embeddings_db.keys())
    db_embeddings = np.array([embeddings_db[player] for player in player_names])
    
    cap = cv2.VideoCapture(video_source)
    
    last_time = time.time()
    last_call_time = 0  # Für Cooldown
    last_match = None  # Für Deduplizierung
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        current_time = time.time()
        if current_time - last_time >= 0.2:  # 5 FPS (reduzierte Rate)
            start_proc = time.time()  # Profiling
            threading.Thread(target=process_frame_with_ocr, args=(frame,)).start()
            # process_frame_with_ocr(frame)
            faces = app.get(frame)
            
            for face in faces:
                embedding = face.embedding.reshape(1, -1)
                
                # Vektorisierte Cosine-Similarity (schneller)
                similarities = 1 - cdist(embedding, db_embeddings, metric='cosine')[0]
                if len(similarities) == 0:
                    continue
                
                best_idx = np.argmax(similarities)
                score = similarities[best_idx]
                
                if score > threshold:
                    best_match = player_names[best_idx]
                    
                    # Nur bei Änderung und Cooldown API aufrufen
                    if best_match != last_match and current_time - last_call_time >= 1:
                        last_match = best_match
                        last_call_time = current_time
                        
                        # Asynchroner API-Call (non-blocking)
                        threading.Thread(target=grafik_overlay_api_call, args=(best_match,)).start()
            
            last_time = current_time
            proc_time = time.time() - start_proc
            print(f"Frame verarbeitet in {proc_time:.2f}s")  # Logging für Debugging
    
    cap.release()


def process_frame_with_ocr(input_frame):
    try:  
        # Variablen für die Cropping Dimensions (in Px)
        y=51
        x=212
        h=99
        w=403
            
        cropped_frame = input_frame[y:y+h, x:x+w]        
        cv2.imshow("FRAME", cropped_frame)
    except Exception as e:
        print(f"Fehler beim OCR Processing: {e}")      


# ---------- API FUNKTIONEN  ----------
def grafik_overlay_api_call(name):
    try:
        response = requests.post(API_URL, json={'text': name})
        print(f"API-Aufruf: {name} (Status: {response.status_code})")
    except Exception as e:
        print(f"Fehler beim API-Aufruf: {e}")



if __name__ == "__main__":
    real_time_api_update(10, 0.65)

