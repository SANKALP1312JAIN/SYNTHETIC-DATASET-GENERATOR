"use client";
import { useState, ChangeEvent, FormEvent } from "react"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001"

const domains = [
  { value: "healthcare", label: "Healthcare" },
  { value: "finance", label: "Finance" },
  { value: "retail", label: "Retail" },
  { value: "custom", label: "Custom" },
]
const privacyLevels = [
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" },
]
const outputFormats = [
  { value: "csv", label: "CSV" },
  { value: "json", label: "JSON" },
  { value: "xlsx", label: "Excel" },
]

export default function GenerateSyntheticDataPage() {
  const [file, setFile] = useState<File | null>(null)
  const [domain, setDomain] = useState("healthcare")
  const [privacy, setPrivacy] = useState("high")
  const [numSamples, setNumSamples] = useState(100)
  const [outputFormat, setOutputFormat] = useState("csv")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    setDownloadUrl(null)
    if (!file) {
      setError("Please upload a dataset file.")
      return
    }
    setLoading(true)
    try {
      const formData = new FormData()
      formData.append("file", file)
      formData.append("domain", domain)
      formData.append("privacy_level", privacy)
      formData.append("num_samples", numSamples.toString())
      formData.append("output_format", outputFormat)

      const response = await fetch(`${API_URL}/generate-synthetic-data`, {
        method: "POST",
        body: formData,
      })
      const data = await response.json()
      if (!response.ok) {
        setError(data.error || "Failed to generate synthetic data.")
        setLoading(false)
        return
      }
      if (data.download_url) {
        setDownloadUrl(`${API_URL}${data.download_url}`)
      } else {
        setError("No download link provided.")
      }
    } catch (err: any) {
      setError(err.message || "An error occurred.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-xl mx-auto py-10">
      <h1 className="text-2xl font-bold mb-6">Generate Synthetic Data</h1>
      <form onSubmit={handleSubmit} className="space-y-6 bg-white p-6 rounded shadow">
        <div>
          <label className="block font-medium mb-2">Upload Dataset (CSV, Excel, etc.)</label>
          <input type="file" accept=".csv,.xlsx,.xls,.json,.parquet,.tsv,.txt" onChange={handleFileChange} className="block w-full" />
        </div>
        <div>
          <label className="block font-medium mb-2">Domain</label>
          <select value={domain} onChange={e => setDomain(e.target.value)} className="w-full border rounded px-3 py-2">
            {domains.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
          </select>
        </div>
        <div>
          <label className="block font-medium mb-2">Privacy Level</label>
          <select value={privacy} onChange={e => setPrivacy(e.target.value)} className="w-full border rounded px-3 py-2">
            {privacyLevels.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
          </select>
        </div>
        <div>
          <label className="block font-medium mb-2">Number of Samples</label>
          <input type="number" min={10} max={10000} value={numSamples} onChange={e => setNumSamples(Number(e.target.value))} className="w-full border rounded px-3 py-2" />
        </div>
        <div>
          <label className="block font-medium mb-2">Output Format</label>
          <select value={outputFormat} onChange={e => setOutputFormat(e.target.value)} className="w-full border rounded px-3 py-2">
            {outputFormats.map(f => <option key={f.value} value={f.value}>{f.label}</option>)}
          </select>
        </div>
        <button type="submit" disabled={loading} className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:bg-gray-400">
          {loading ? "Generating..." : "Generate Synthetic Data"}
        </button>
        {error && <div className="text-red-600 mt-2">{error}</div>}
        {downloadUrl && (
          <a href={downloadUrl} className="block mt-4 text-blue-600 underline" download>
            Download Synthetic Data
          </a>
        )}
      </form>
    </div>
  )
} 