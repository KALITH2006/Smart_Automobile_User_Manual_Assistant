"use client";

import { useRef, useState } from "react";
import { api, Manual } from "../lib/api";

export default function FileUpload({
  onUploaded,
}: {
  onUploaded: (manual: Manual) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [vehicleName, setVehicleName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFile(file: File) {
    setError(null);
    setUploading(true);
    try {
      const manual = await api.uploadManual(
        file,
        vehicleName.trim() || file.name.replace(/\.pdf$/i, "")
      );
      onUploaded(manual);
      setVehicleName("");
    } catch (e: any) {
      setError(e.message ?? "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="border border-dashed border-slate-600 rounded-xl p-4 bg-slate-900/40">
      <p className="text-sm text-slate-300 mb-2 font-medium">Upload a vehicle manual (PDF)</p>
      <input
        type="text"
        placeholder="Vehicle name (e.g. 2024 Honda Civic)"
        value={vehicleName}
        onChange={(e) => setVehicleName(e.target.value)}
        className="w-full mb-2 px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-brand-500"
      />
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
      <button
        onClick={() => inputRef.current?.click()}
        disabled={uploading}
        className="w-full px-3 py-2 rounded-lg bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-sm font-medium transition-colors"
      >
        {uploading ? "Processing manual…" : "Choose PDF"}
      </button>
      {error && <p className="text-xs text-red-400 mt-2">{error}</p>}
    </div>
  );
}
