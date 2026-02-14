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

GROQ_API_KEY = "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"
NOTION_TOKEN = "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"
NOTION_DATABASE_ID = "247d802e23e080499fecc25b1f4b5cb2"
batch_model= WhisperModel("large-v3-turbo", device="cpu", compute_type= "int8")
small_model= WhisperModel("tiny.en", device= "cpu", compute_type="int8")
class Ultimate():
    def __init__(self):
        self.win=tk.Tk()
        self.win.title("Voice Transcription to Notion ")
        CLASS_OPTIONS= ["Food Science 150", "CHEM 111", "CICS 160", "Others"]
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
            self.button.config(text= " START RECORDING ")
            print("Recording Stopped, Summarizing and saving to Notion")
            threading.Thread(target= self.end_session, daemon= True ).start()
        
    def sound_settings(self):
        samplerate=16000
        block_duration= 0.5
        
        frames_per_block= int(samplerate*block_duration)   
        self.stream =sd.InputStream(channels=1, samplerate= samplerate, blocksize= frames_per_block, callback= self.audio_transcription)
    
    def audio_transcription(self, indata, frames, times, status):
        if self.is_recording:
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
                    segment,info= small_model.transcribe(audio_data,beam_size=2, vad_filter= True, word_timestamps= False)
                    
                    for s in segment:
                        if s.text.strip():
                            self.update_read(s.text)
                            self.transcript+= s.text +" "
            except queue.Empty:
                continue 
        self.stream.stop()
        self.stream.close()
    
    def update_read(self,text):
        self.text_transcript.config(state= "normal")
        self.text_transcript.insert(tk.END, text+ " ")
        self.text_transcript.see(tk.END)
        self.text_transcript.config(state= "disabled")
       
    def generate_summary(self):
        print("Sending transcript to summarize!") 
        self.status_label.config(text="GENERATING SUMMARY", fg="yellow")
        try:
            completion = self.client.chat.completions.create(model= "meta-llama/llama-4-maverick-17b-128e-instruct", messages=[
                {
                    "role": "system",
                    "content": "You are a specialized assistant whose main purpose is to summarize the transcript exactly by taking all the major points and minor points which can be tested on for a quiz or an exam. Also give these points precisely and avoid any grammatical errors and use context from previous sentences to summarize more accuratelty. Research using the internet, to get precise points and fix any errors that may have been encountered during transcription."
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
                    "content": "You are a specialized assistant whose main purpose is to summarize the transcript exactly by taking all the major points and minor points which can be tested on for a quiz or an exam. Also give these points precisely and avoid any grammatical errors and use context from previous sentences to summarize more accuratelty. Research using the internet, to get precise points and fix any errors that may have been encountered during transcription."
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
        audio_complete= np.concatenate(self.full_data).flatten()
        segments,info= batch_model.transcribe(audio_complete, beam_size= 5)
        self.transcript= "".join([s.text for s in segments])
        summ= self.generate_summary()
        self.update_read(" --- AI SUMMARY ---  " + summ)
        try:
            self.notion(summ)
        except:
            self.save_locally(summ)
        
    def seperate_text(self,text,size=2000):
        return[text[i:i+size] for i in range (0,len(text),size)]
    
    
    def notion(self,summ):
        now=datetime.now().isoformat()
        self.status_label.config(text="SAVED TO NOTION", fg="green")
        print("Push to Notion..")
        url= "https://api.notion.com/v1/pages"
        headers={
            "Authorization" : f"Bearer {NOTION_TOKEN}",
            "Content-Type" : "application/json",
            "Notion-Version" : "2025-09-03"
        }
        chunks= self.seperate_text(self.transcript)
        blocks = [
            {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "AI Summary"}}]}},
            {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": summ}}]}},
            {"object": "block", "type": "divider", "divider": {}},
            {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"text": {"content": "Full Transcript"}}]}}
        ]
        for c in chunks:
            blocks.append({
                "object": "block",
                "type":"paragraph",
                "paragraph" : {'rich_text' :[{"text" :{"content":c}}] }
            })
            
        payload= {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "properties":{"Date":{"date" :{"start":now}},
               "Name": {"title": [{"text": {"content": self.entry.get() or "Lecture Notes"}}]},
                "Class": {"select": {"name": self.class_dropdown.get()} }
                 
            },
           "children": blocks
        }
        try:
            response= requests.post(url, headers=headers, json=payload)
            if response.status_code== 200:
                print("Your summarized data is sent to Notion!")
                self.update_read("SUCCESSFULLY SAVED TO NOTION ")
            else:
                print("Error!")
        except Exception as e:
            print(f"Notion connection failed {e}")  
             
    def save_locally(self,summ):
        with open(self.entry.get() + ".txt", "w")as d:
            d.write("SUMMARY: " + summ)
            d.write("TRANSCRIPT: "+ self.transcript)
    
    
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category= UserWarning, module= 'multiprocessing.resource_tracker')
    x=Ultimate()


