import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import threading
import queue
from groq import Groq
import warnings
import requests
from datetime import datetime
import mlx_whispet

GROQ_API_KEY = "ENTER YOUR API KEY HERE "
NOTION_TOKEN = "ENTER YOUR TOKEN KEY HERE "
NOTION_DATABASE_ID = "ENTER YOUR DATABASE KEY HERE"
batch_model="openai/whisper-large-v3-turbo"
small_model= WhisperModel("tiny.en", device= "cpu", compute_type="int8")
# get a login token from huggingface_hub to get the process to run much faster.
class Ultimate():
    def __init__(self):
        self.win=tk.Tk()
        self.win.title("Voice Transcription to Notion ")
        CLASS_OPTIONS= ["Food Science 150", "CHEM 111", "CICS 160", "General"]
        self.class_dropdown= ttk.Combobox(self.win, values= CLASS_OPTIONS)
        self.class_dropdown.pack()
        tk.Label(self.win, text= "Topic/Lesson Name" , font= ("Arial", 10, "bold"), bg= 'black', fg='white').pack(anchor="w", padx=15, pady=(10,0))
        self.entry= tk.Entry(self.win, font=("Arial", 12))
        self.entry.pack(fill="x", padx=15, pady=5)
        self.button= tk.Button(self.win, text= "START RECORDING", command= self.recording, padx=20, pady=10, activebackground= 'white', activeforeground='black')
        self.button.pack(pady=20)
        tk.Label(self.win, text="Recorded Transcript: ", font= ("Arial", 10, "bold"), bg= 'black', fg='white').pack(anchor="w", padx=15, pady=(10,0))
        self.text_transcript= scrolledtext.ScrolledText(self.win, font=("Arial", 10), bg="#111", fg="#00FF00", height= 15, state='disabled')
        self.text_transcript.pack(fill= "both", expand= True, padx= 15, pady= 10)
        self.is_recording = False
        self.status_label= tk.Label(self.win, text=" READY", fg="green", bg="black", font=("Arial", 10, "bold"))
        self.status_label.pack(side="bottom", fill="x")
        self.queue= queue.Queue()
        self.audio= []
        self.full_data=[]
        self.transcript=""
        self.client= Groq(api_key=GROQ_API_KEY)
        self.win.protocol("WM_DELETE_WINDOW", self.closing)
        
        self.win.mainloop()
        
    def recording(self):
        if self.is_recording== False:
            self.is_recording= True
        
            self.button.config(text= "STOP RECORDING")
            thread= threading.Thread(target= self.run_ai_logic, daemon= True).start()
            print("Starting Recording")
            self.status_label.config(text="RECORDING", fg="red")
        else:
            self.is_recording= False
            self.button.config(text="WAIT...", state="disabled")
            print("Recording Stopped, Summarizing and saving to Notion")
            threading.Thread(target= self.end_session, daemon= True ).start()
        
    def sound_settings(self):
        samplerate=16000
        block_duration= 0.5
   
        frames_per_block= int(samplerate*block_duration)   
        self.stream =sd.InputStream(channels=1, samplerate= samplerate, blocksize= frames_per_block, device= 0, callback= self.audio_transcription)
    
    def audio_transcription(self, indata, frames, times, status):
        if self.is_recording:
            volume_norm = np.linalg.norm(indata) * 20
            if volume_norm > 0.1:
                print(f" LEVEL: {'█' * int(min(volume_norm, 50))}")
            else:
                print(" SILENCE (Zero Data)")
            self.queue.put(indata.copy())
            self.full_data.append(indata.copy())
            
    def run_ai_logic(self):
        self.sound_settings()
        self.stream.start()
        audio_cache=[]
        
        while self.is_recording== True:
            try:
                block= self.queue.get(timeout= 1.0)
                audio_cache.append(block)
                if len(audio_cache)>=4:
                    audio_data= np.concatenate(audio_cache).flatten()
                    audio_cache=[]
                    segment,info= small_model.transcribe(audio_data,beam_size=2, vad_filter= False, word_timestamps= False)
                    
                    for s in segment:
                        if s.text.strip():
                            self.update_read(s.text)
                            self.transcript+= s.text +" "
            except queue.Empty:
                continue 
        self.stream.stop()
        self.stream.close()
    
    def update_read(self,text):
        def _update():
            self.text_transcript.config(state= "normal")
            self.text_transcript.insert(tk.END, text+ " ")
            self.text_transcript.see(tk.END)
            self.text_transcript.config(state= "disabled")
        self.win.after(0, _update)
       
    def generate_summary(self):
        print("Sending transcript to summarize!") 
        self.win.after(0, lambda: self.status_label.config(text="GENERATING SUMMARY", fg="yellow"))
        try:
            completion = self.client.chat.completions.create(model= "meta-llama/llama-4-maverick-17b-128e-instruct", messages=[
                {
                    "role": "system",
                    "content": "You are a specialized assistant whose main purpose is to summarize the transcript exactly by taking all the major points and minor points which can be tested on for a quiz or an exam. Also give these points precisely and avoid any grammatical errors and use context from previous sentences to summarize more accuratelty. Research using the internet, to get precise points and fix any errors that may have been encountered during transcription. Expand on some important points to give an explanation to deepen understanding."
                },
                
            
            {
                "role": "user",
                "content": self.transcript
                
            }
            ],
            temperature= 0.6,
            max_tokens= 1250)
            return completion.choices[0].message.content
        except:
            completion = self.client.chat.completions.create(model= "llama-3.3-70b-versatile", messages=[
                {
                    "role": "system",
                    "content": "You are a specialized assistant whose main purpose is to summarize the transcript exactly by taking all the major points and minor points which can be tested on for a quiz or an exam. Also give these points precisely and avoid any grammatical errors and use context from previous sentences to summarize more accuratelty. Research using the internet, to get precise points and fix any errors that may have been encountered during transcription. Expand on some important points to give an explanation to deepen understanding."
                },
                
            
            {
                "role": "user",
                "content": self.transcript   
            }
            ],
            temperature= 0.6,
            max_tokens= 1500)
            return completion.choices[0].message.content
    def closing(self):
        self.is_recording= False
        if hasattr(self,'stream'):
            self.stream.stop()
            self.stream.close()
        self.win.destroy()
            
    def end_session(self):
        if not self.full_data:
            print("⚠ Error: No audio data captured. Check macOS Mic Permissions.")
            self.win.after(0, lambda: self.status_label.config(text="⚠ NO AUDIO DATA", fg="red"))
            return
        audio_complete= np.concatenate(self.full_data).flatten().astype(np.float32)
        batch_model = "mlx-community/whisper-large-v3-turbo"
        segments= mlx_whisper.transcribe(audio_complete,path_or_hf_repo= batch_model, verbose= False)
        self.transcript= segments['text']
        summ= self.generate_summary()
        self.update_read(" --- AI SUMMARY ---  " + summ)
        try:
            self.notion(summ)
        except:
            self.save_locally(summ)
        finally:
            self.full_data=[]
            self.audio=[]
        
    def seperate_text(self,text,size=1000):
        if not text: return ["No content."]
        return[text[i:i+size] for i in range (0,len(text),size)]
    def notion(self, summ):
        now = datetime.now().isoformat()
        self.win.after(0, lambda: self.status_label.config(text="SAVING SUMMARY...", fg="yellow"))
        print("Pushing to Notion...")
        
        url = "https://api.notion.com/v1/pages"
        headers = {
            "Authorization": f"Bearer {NOTION_TOKEN}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28" 
        }

        # 1. Chunk the summary correctly
        summary_chunks = self.seperate_text(summ, size=1900)
        
        blocks = [
            {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "✨ AI Summary"}}]}}
        ]
        
        for s_chunk in summary_chunks:
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": s_chunk}}]}
            })

        payload = {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "properties": {
                "Date": {"date": {"start": now}},
                "Name": {"title": [{"text": {"content": self.entry.get() or "Lecture Notes"}}]},
                "Class": {"select": {"name": self.class_dropdown.get() or "General"}}
            },
            "children": blocks
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                page_id = response.json().get("id")
                print(f"✅ Summary Saved! ID: {page_id}")
                
                # 3. Start the background thread to sync the massive transcript
                threading.Thread(target=self.sync_transcript, args=(page_id, headers), daemon=True).start()
                self.update_read("\n--- SUCCESS: DATA SENT TO NOTION ---")
            else:
                print(f"❌ NOTION API ERROR {response.status_code}:")
                error_data = response.json()
                print(f"Message: {error_data.get('message')}")
                self.save_locally(summ)
        except Exception as e:
            print(f"📡 Notion connection failed: {e}")
            self.save_locally(summ) 
            
    def sync_transcript(self, page_id, headers):
        """Appends full transcript in batches of 100 blocks."""
        self.win.after(0, lambda: self.status_label.config(text="SYNCING FULL TEXT...", fg="blue"))
        append_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        transcript_chunks = self.seperate_text(self.transcript)
        
        all_blocks = [
            {"object": "block", "type": "divider", "divider": {}},
            {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "📝 Full Transcript"}}]}}
        ]
        for c in transcript_chunks:
            all_blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": c}}]}})

        for i in range(0, len(all_blocks), 100):
            batch = all_blocks[i:i+100]
            requests.patch(append_url, headers=headers, json={"children": batch}, timeout=20)
        
        self.win.after(0, lambda: self.status_label.config(text="FULLY SAVED", fg="green"))
        self.win.after(0, lambda: self.button.config(state="normal", text="START RECORDING"))
             
    def save_locally(self,summ):
        with open(self.entry.get() + ".txt", "w")as d:
            d.write("SUMMARY: " + summ)
            d.write("TRANSCRIPT: "+ self.transcript)
    
    
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category= UserWarning, module= 'multiprocessing.resource_tracker')
    x=Ultimate()





