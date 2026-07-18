import os
import shutil
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from DetectNP import load_registry, save_registry, canonicalize_plate
from PIL import Image, ImageTk

class RegistryTab(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="#090a0f", **kwargs)
        
        # Fields variables
        self.plate_var = ctk.StringVar()
        self.owner_name_var = ctk.StringVar()
        self.phone_var = ctk.StringVar()
        self.notes_var = ctk.StringVar()
        self.photo_path = ctk.StringVar()
        self.photo_preview_img = None

        # Split Layout: Left list, Right editing form
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self._build_left_list()
        self._build_right_form()
        self.refresh_list()

    def _build_left_list(self):
        left_fr = ctk.CTkFrame(self, fg_color="#030408", border_width=1, border_color="#1e293b")
        left_fr.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
        left_fr.grid_rowconfigure(1, weight=1)
        left_fr.grid_columnconfigure(0, weight=1)

        # Top search container
        search_fr = ctk.CTkFrame(left_fr, fg_color="transparent")
        search_fr.grid(row=0, column=0, sticky="ew", padx=15, pady=10)
        search_fr.grid_columnconfigure(1, weight=1)

        lbl = ctk.CTkLabel(search_fr, text="Vehicle Database", font=("Inter", 14, "bold"), text_color="#f8fafc")
        lbl.grid(row=0, column=0, sticky="w", padx=(0, 15))

        # Search box
        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", lambda *_: self.refresh_list())
        search_bar = ctk.CTkEntry(search_fr, placeholder_text="Search by plate or owner...",
                                  textvariable=self.search_var, fg_color="#0d111a", border_color="#1e293b")
        search_bar.grid(row=0, column=1, sticky="ew")

        # Modern ttk Treeview for data listing
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#0d111a", fieldbackground="#0d111a", foreground="#e2e8f0",
                        bordercolor="#1e293b", rowheight=28, gridlinescolor="#1e293b")
        style.map("Treeview", background=[("selected", "#0ea5e9")], foreground=[("selected", "#ffffff")])
        
        self.tree = ttk.Treeview(left_fr, columns=("plate", "owner", "phone"), show="headings")
        self.tree.heading("plate", text="Plate Number")
        self.tree.heading("owner", text="Owner Name")
        self.tree.heading("phone", text="Phone Number")
        
        self.tree.column("plate", width=120)
        self.tree.column("owner", width=160)
        self.tree.column("phone", width=120)
        
        self.tree.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))
        self.tree.bind("<<TreeviewSelect>>", self.on_select_record)

    def _build_right_form(self):
        right_fr = ctk.CTkFrame(self, fg_color="#030408", border_width=1, border_color="#1e293b")
        right_fr.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
        right_fr.grid_columnconfigure(0, weight=1)

        lbl = ctk.CTkLabel(right_fr, text="Registry Record Details", font=("Inter", 14, "bold"), text_color="#f8fafc")
        lbl.grid(row=0, column=0, sticky="w", padx=15, pady=10)

        fields = [("License Plate", self.plate_var), ("Owner Name", self.owner_name_var), 
                  ("Contact Phone", self.phone_var), ("Notes / Memo", self.notes_var)]
        
        for idx, (label_txt, var) in enumerate(fields, start=1):
            ctk.CTkLabel(right_fr, text=label_txt, font=("Inter", 12), text_color="#94a3b8").grid(row=idx*2-1, column=0, sticky="w", padx=15, pady=(5, 0))
            ctk.CTkEntry(right_fr, textvariable=var, fg_color="#0d111a", border_color="#1e293b").grid(row=idx*2, column=0, sticky="ew", padx=15, pady=(0, 5))

        # Photo section
        photo_row = len(fields)*2 + 1
        ctk.CTkLabel(right_fr, text="Owner Photo Preview", font=("Inter", 12), text_color="#94a3b8").grid(row=photo_row, column=0, sticky="w", padx=15, pady=(5, 0))
        
        self.photo_lbl = ctk.CTkLabel(right_fr, text="No photo registered", font=("Inter", 11), fg_color="#0d111a", height=80)
        self.photo_lbl.grid(row=photo_row+1, column=0, sticky="ew", padx=15, pady=5)
        
        btn_photo = ctk.CTkButton(right_fr, text="Browse Photo", command=self.on_browse_photo, fg_color="#0f172a", border_color="#1e293b", border_width=1)
        btn_photo.grid(row=photo_row+2, column=0, sticky="ew", padx=15, pady=5)

        # Action buttons CRUD
        btn_panel = ctk.CTkFrame(right_fr, fg_color="transparent")
        btn_panel.grid(row=photo_row+3, column=0, sticky="ew", padx=15, pady=15)
        
        ctk.CTkButton(btn_panel, text="New", command=self.on_new_record, width=60, fg_color="#0f172a", border_color="#1e293b", border_width=1).pack(side="left", padx=2)
        ctk.CTkButton(btn_panel, text="Save", command=self.on_save_record, width=80, fg_color="#0ea5e9").pack(side="right", padx=2)
        ctk.CTkButton(btn_panel, text="Delete", command=self.on_delete_record, width=60, fg_color="#ef4444").pack(side="right", padx=2)

    def refresh_list(self):
        self.tree.delete(*self.tree.get_children())
        try:
            df = load_registry()
        except Exception:
            return
        
        query = self.search_var.get().strip().lower()
        for _, row in df.iterrows():
            plate = str(row.get("plate", ""))
            owner = str(row.get("owner_name", ""))
            phone = str(row.get("phone", ""))
            
            if query and query not in plate.lower() and query not in owner.lower():
                continue
                
            self.tree.insert("", "end", values=(plate, owner, phone))

    def on_select_record(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        try:
            df = load_registry()
            row = df[df["plate_norm"] == canonicalize_plate(vals[0])].iloc[0]
        except Exception:
            return
        
        self.plate_var.set(str(row.get("plate", "")))
        self.owner_name_var.set(str(row.get("owner_name", "")))
        self.phone_var.set(str(row.get("phone", "")))
        self.notes_var.set(str(row.get("notes", "")))
        photo = str(row.get("photo", ""))
        self.photo_path.set(photo)
        self._update_photo_preview(photo)

    def _update_photo_preview(self, path):
        if path and os.path.isfile(path):
            try:
                img = Image.open(path)
                img.thumbnail((150, 80))
                self.photo_preview_img = ImageTk.PhotoImage(img)
                self.photo_lbl.configure(image=self.photo_preview_img, text="")
            except Exception:
                self.photo_lbl.configure(image=None, text="Error loading photo")
        else:
            self.photo_lbl.configure(image=None, text="No photo registered")

    def on_browse_photo(self):
        p = filedialog.askopenfilename(title="Select owner photo", filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")])
        if p:
            self.photo_path.set(p)
            self._update_photo_preview(p)

    def on_new_record(self):
        self.plate_var.set("")
        self.owner_name_var.set("")
        self.phone_var.set("")
        self.notes_var.set("")
        self.photo_path.set("")
        self.photo_lbl.configure(image=None, text="No photo registered")

    def on_save_record(self):
        plate = self.plate_var.get().strip()
        if not plate:
            messagebox.showerror("Error", "Plate number is required")
            return
        
        try:
            df = load_registry()
        except Exception:
            df = pd.DataFrame(columns=["plate", "owner_name", "phone", "notes", "photo", "plate_norm"])

        src_photo = self.photo_path.get()
        dest_photo = src_photo
        
        # Copy photo to local project files if it's external
        if src_photo and os.path.isfile(src_photo) and "owners" not in src_photo:
            owners_dir = os.path.join("runs", "anpr_yolo", "owners")
            os.makedirs(owners_dir, exist_ok=True)
            canon = canonicalize_plate(plate).replace("-", "_").replace(".", "_")
            ext = os.path.splitext(src_photo)[1] or ".jpg"
            dest = os.path.join(owners_dir, f"{canon}{ext}")
            shutil.copy2(src_photo, dest)
            dest_photo = dest

        new_row = {
            "plate": plate,
            "owner_name": self.owner_name_var.get().strip(),
            "phone": self.phone_var.get().strip(),
            "notes": self.notes_var.get().strip(),
            "photo": dest_photo,
            "plate_norm": canonicalize_plate(plate)
        }

        key = canonicalize_plate(plate)
        mask = df["plate_norm"] == key if "plate_norm" in df.columns else pd.Series([False]*len(df))
        
        if mask.any():
            for col, val in new_row.items():
                df.loc[mask, col] = val
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        save_registry(df)
        self.refresh_list()
        messagebox.showinfo("Saved", f"Record saved for {plate}")

    def on_delete_record(self):
        plate = self.plate_var.get().strip()
        if not plate:
            return
        if not messagebox.askyesno("Delete", f"Are you sure you want to delete {plate}?"):
            return
        
        try:
            df = load_registry()
            key = canonicalize_plate(plate)
            df = df[df["plate_norm"] != key]
            save_registry(df)
            self.on_new_record()
            self.refresh_list()
        except Exception as e:
            messagebox.showerror("Error", str(e))
