"""
LPGA Player Face Recognition System
Updated for Python 3.13 with simplified output
"""

import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import urllib.request
import argparse
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
import time
from tqdm import tqdm
from scipy.spatial.distance import cdist
from PIL import Image
import pytesseract
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


# Configuration
@dataclass
class Config:
    """Configuration for the face recognition system"""
    dataset_dir: str = "dataset"
    embeddings_file: str = "player_embeddings.pkl"
    model_name: str = "buffalo_l"  # Latest model: buffalo_l (balanced), buffalo_sc (accurate but slower)
    threshold: float = 0.65
    processing_fps: float = 5.0  # Process every N frames per second
    cooldown_seconds: float = 1.0  # Minimum time between name changes
    output_url: Optional[str] = None  # Will be used for web server integration


# Liste der Top LPGA-Spielerinnen (erweitert basierend auf 2025-Rankings)
PLAYERS = [
    "Casandra Alexander", "Helen Alfredsson", "Carmen Alonso", "Ela Anacona",
    "Pajaree Anannarukarn", "April Angurasaranee", "Tiffany Arafi", "Amanda Ard",
    "Kajsa Arwefjall", "Aditi Ashok", "Aloysa Atienza", "Miriam Ayora",
    "Pia Babnik", "Lianna Bailey", "Jess Baker", "Hitaashee Bakshi",
    "Susan Bamford", "Ana Belac", "Rosie Belsham", "Kelsey Bennett",
    "Laura Beveridge", "Gudrun Bjorgvinsdottir", "Carly Booth", "Celine Borge",
    "Annika Borrelli", "Justice Bosio", "Amy Boulden", "Celine Boutier",
    "Vanessa Bouvet", "Lyn Brand", "Stacy Bregman", "Becky Brewerton",
    "Helen Briem", "Sofie Bringner", "Nicole Broch Estrup", "Samantha Bruce",
    "Ashleigh Buhai", "Kan Bunnabodee", "Hannah Burke", "Maxine Burton",
    "Sara Byrne", "Virginia Elena Carta", "Anne-Lise Caudal", "Tiffany Chan",
    "Trichat Cheenglab", "Camille Chevalier", "Pamela Chugg MBE", "Carlota Ciganda",
    "Gemma Clews", "Holly Clyburn", "Pasqualle Coffa", "Selena Costabile",
    "Olivia Cowan", "Gabriella Cowley", "Diksha Dagar", "Daniela Darquea",
    "Klara Davidson Spilkova", "Laura Davies", "Rosie Davies", "Hayley Davis",
    "Ana Dawson", "Marie Laure De Lorenzi", "Manon De Roey", "Perrine Delacour",
    "Jane Denman", "Megan Dennis", "Teresa Diez Moliner", "Annabel Dimmock",
    "Ginnie Ding", "Amandeep Drall", "Gemma Dryburgh", "Danielle Du Toit",
    "Mia Eales-Smith", "Margaret Egan", "Karstin Ehrnlund", "Marina Escobar Domingo",
    "Jodi Ewart Shadoff", "Alessandra Fanali", "Natasha Fear", "Fatima Fernandez Cano", 
    "Blanca Fernandez", "Michaela Finn", "Cecilie Finne-Ipsen", "Moa Folke", 
    "Dorthea Forbrigd", "Jane Forrest", "Michelle Forsland", "Alexandra Forsterling", 
    "Anna Foster", "Moa Fridell", "Laura Fuenfstueck", "Reina Fujikawa", "Annabell Fuller", 
    "Cara Gainer", "Sandra Gal", "Luna Sobron Galmes", "Nicole Garcia", "Amelia Garvey", 
    "Sarah Gee", "Verena Gimmy", "Eleanor Givens", "Laura Gomez Ruiz", "Alice Gotbring", 
    "Ellie Gower", "Linn Grant", "Emma Grechi", "Chris Green", "Hannah Gregg", "Christine Griffith", 
    "Natascha Grossschadl-fink", "Nataliya Guseva", "Sophie Gustafson", "Maha Haddioui",
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


class PlayerRecognitionSystem:
    """Main class for LPGA player face recognition"""
    
    def __init__(self, config: Config):
        self.config = config
        self.app = None
        self.embeddings_db: Dict[str, np.ndarray] = {}
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize InsightFace model with latest version"""
        print(f"Initializing InsightFace model: {self.config.model_name}")
        self.app = FaceAnalysis(
            name=self.config.model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("Model initialized successfully")
    
    def build_dataset_with_progress(self, start_index: int = 0):
        """Build dataset by downloading and processing player images"""
        os.makedirs(self.config.dataset_dir, exist_ok=True)
        
        # Load progress if exists
        progress_file = Path('progress.json')
        if progress_file.exists() and start_index == 0:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                start_index = progress.get('last_player_index', 0)
                print(f"Resuming from player index: {start_index}")
        
        for idx, player in enumerate(PLAYERS[start_index:], start=start_index):
            print(f"\nProcessing player {idx + 1}/{len(PLAYERS)}: {player}")
            player_dir = os.path.join(self.config.dataset_dir, player.replace(" ", "_"))
            os.makedirs(player_dir, exist_ok=True)
            
            if os.listdir(player_dir):
                print(f"Skipping {player} - already exists")
                continue
            
            # Here you would integrate your image search/download logic
            # For now, this is a placeholder
            print(f"Note: Image download not implemented - add your source here")
            
            # Save progress
            with open(progress_file, 'w') as f:
                json.dump({'last_player_index': idx + 1}, f)
    
    def generate_embeddings(self):
        """Generate face embeddings for all players in dataset"""
        print("Generating embeddings for player dataset...")
        embeddings = {}
        
        dataset_path = Path(self.config.dataset_dir)
        if not dataset_path.exists():
            print(f"Dataset directory not found: {self.config.dataset_dir}")
            return
        
        for player in tqdm(PLAYERS, desc="Processing players"):
            player_dir = dataset_path / player.replace(" ", "_")
            
            if not player_dir.exists() or not list(player_dir.glob("*.jpg")):
                continue
            
            player_embeddings = []
            for img_file in player_dir.glob("*.jpg"):
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                faces = self.app.get(img)
                if len(faces) > 0:
                    player_embeddings.append(faces[0].embedding)
            
            if player_embeddings:
                # Average embedding for robustness
                embeddings[player] = np.mean(player_embeddings, axis=0)
                print(f"✓ {player}: {len(player_embeddings)} embeddings")
        
        if embeddings:
            with open(self.config.embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"\n✓ Saved embeddings for {len(embeddings)} players")
        else:
            print("✗ No embeddings generated")
    
    def load_embeddings(self) -> bool:
        """Load pre-generated embeddings"""
        embeddings_path = Path(self.config.embeddings_file)
        if not embeddings_path.exists():
            print(f"Embeddings file not found: {self.config.embeddings_file}")
            print("Run with --generate_embeddings first")
            return False
        
        with open(embeddings_path, 'rb') as f:
            self.embeddings_db = pickle.load(f)
        
        print(f"✓ Loaded embeddings for {len(self.embeddings_db)} players")
        return True
    
    def recognize_face(self, face_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face by comparing against database
        Returns: (player_name, confidence_score) or (None, 0.0)
        """
        if not self.embeddings_db:
            return None, 0.0
        
        embedding = face_embedding.reshape(1, -1)
        player_names = list(self.embeddings_db.keys())
        db_embeddings = np.array([self.embeddings_db[p] for p in player_names])
        
        # Vectorized cosine similarity
        similarities = 1 - cdist(embedding, db_embeddings, metric='cosine')[0]
        
        best_idx = np.argmax(similarities)
        score = similarities[best_idx]
        
        if score > self.config.threshold:
            return player_names[best_idx], score
        return None, score
    
    def output_recognition(self, player_name: str, confidence: float):
        """
        Output recognized player name
        This is where web server integration will happen
        """
        timestamp = time.strftime("%H:%M:%S")
        output_msg = f"[{timestamp}] Recognized: {player_name} (confidence: {confidence:.2f})"
        print(output_msg)
        
        # TODO: Future web server integration
        # if self.config.output_url:
        #     try:
        #         response = requests.post(
        #             self.config.output_url,
        #             json={'player': player_name, 'confidence': confidence, 'timestamp': timestamp}
        #         )
        #     except Exception as e:
        #         print(f"Error sending to server: {e}")
    
    def real_time_recognition(self, video_source: int = 0):
        """
        Real-time face recognition from video source
        Simplified version that just outputs player names
        """
        if not self.load_embeddings():
            return
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Cannot open video source {video_source}")
            return
        
        print(f"\nStarting real-time recognition from source {video_source}")
        print(f"Threshold: {self.config.threshold}")
        print(f"Press 'q' to quit\n")
        
        last_time = time.time()
        last_match = None
        last_output_time = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Process at configured FPS rate
            if current_time - last_time >= (1.0 / self.config.processing_fps):
                faces = self.app.get(frame)
                
                for face in faces:
                    player_name, confidence = self.recognize_face(face.embedding)
                    
                    if player_name:
                        # Only output if name changed and cooldown elapsed
                        if (player_name != last_match and 
                            current_time - last_output_time >= self.config.cooldown_seconds):
                            
                            self.output_recognition(player_name, confidence)
                            last_match = player_name
                            last_output_time = current_time
                            
                            # Optional: Draw on frame for debugging
                            bbox = face.bbox.astype(int)
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                            cv2.putText(frame, f"{player_name}", (bbox[0], bbox[1]-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                last_time = current_time
            
            # Optional: Show frame (comment out for headless operation)
            cv2.imshow('Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")


def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(
        description='LPGA Player Face Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings from dataset
  python golf_player_face_recognition.py --generate_embeddings
  
  # Run real-time recognition on webcam
  python golf_player_face_recognition.py --recognize --source 0
  
  # Run with custom threshold
  python golf_player_face_recognition.py --recognize --source 0 --threshold 0.7
  
  # Build dataset (placeholder - implement your image source)
  python golf_player_face_recognition.py --build_dataset
        """
    )
    
    parser.add_argument('--build_dataset', action='store_true',
                       help='Build player image dataset')
    parser.add_argument('--generate_embeddings', action='store_true',
                       help='Generate face embeddings from dataset')
    parser.add_argument('--recognize', action='store_true',
                       help='Run real-time face recognition')
    parser.add_argument('--source', type=int, default=0,
                       help='Video source (default: 0 for webcam)')
    parser.add_argument('--threshold', type=float, default=0.65,
                       help='Recognition confidence threshold (default: 0.65)')
    parser.add_argument('--model', type=str, default='buffalo_l',
                       choices=['buffalo_l', 'buffalo_sc', 'buffalo_s'],
                       help='InsightFace model to use (default: buffalo_l)')
    parser.add_argument('--output_url', type=str, default=None,
                       help='Web server URL for sending recognition results (future use)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        threshold=args.threshold,
        model_name=args.model,
        output_url=args.output_url
    )
    
    # Initialize system
    system = PlayerRecognitionSystem(config)
    
    # Execute requested operation
    if args.build_dataset:
        system.build_dataset_with_progress()
    elif args.generate_embeddings:
        system.generate_embeddings()
    elif args.recognize:
        system.real_time_recognition(video_source=args.source)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
