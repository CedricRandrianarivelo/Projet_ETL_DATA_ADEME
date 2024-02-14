# -*- coding: utf-8 -*-
"""
Created on Mon Feb 5 10:11:49 2024

@author: cedri
"""

import tkinter as tk
from tkinter import Tk
from tkinter import ttk
from Data_Engineering_Project_ETL import DataEngineeringProject

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Data Engineering Project")
        self.master.geometry("500x300")  # Ajustez la taille de la fenêtre selon vos préférences

        style = ttk.Style()
        style.configure("TButton", padding=10)  # Augmentez la taille du bouton

        self.run_button = ttk.Button(master, text="Run Script", command=self.run_script)
        self.run_button.pack(pady=20)  # Ajoutez un espacement en bas du bouton

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

    def run_script(self):
        project = DataEngineeringProject()
        
        # 1. Récupération des données
        data_api = project.fetch_data_from_api()
        if data_api:
            url_csv = data_api[2]["url"]
            data_csv = project.load_data_from_csv(url_csv)

            # 2. Transformation des données
            data_processed = project.process_data(data_csv)

            # 3. Sauvegarde des données propres
            project.save_cleaned_data(data_processed)
            print("Projet sauvegardé")
            self.result_label.config(text="Script executed successfully.")
        else:
            self.result_label.config(text="Failed to fetch data from API.")


if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
