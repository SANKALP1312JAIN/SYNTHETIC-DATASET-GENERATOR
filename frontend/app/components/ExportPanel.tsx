"use client"

import { Download, FileText, Database, Code } from "lucide-react"
import { useAppContext } from "../context/AppContext"

const exportFormats = [
  {
    id: "csv",
    name: "CSV",
    description: "Comma-separated values",
    icon: FileText,
    color: "bg-green-500",
  },
  {
    id: "parquet",
    name: "Parquet",
    description: "Columnar storage format",
    icon: Database,
    color: "bg-blue-500",
  },
  {
    id: "json",
    name: "JSON",
    description: "JavaScript Object Notation",
    icon: Code,
    color: "bg-purple-500",
  },
]

export default function ExportPanel() {
  const { uploadedFile } = useAppContext()

  const handleExport = async (format: string) => {
    // Mock API call to backend
    try {
      const response = await fetch("/api/export", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          format,
          data: uploadedFile?.data || [],
        }),
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = `synthetic_data.${format}`
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      }
    } catch (error) {
      console.error("Export failed:", error)
      // For demo purposes, create a mock download
      const mockData = uploadedFile?.data || [{ sample: "data" }]
      const dataStr =
        format === "json"
          ? JSON.stringify(mockData, null, 2)
          : mockData.map((row) => Object.values(row).join(",")).join("\n")

      const blob = new Blob([dataStr], { type: "text/plain" })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `synthetic_data.${format}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Export Synthetic Data</h3>

      {!uploadedFile && <p className="text-gray-500 mb-4">Upload and process data first to enable export</p>}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {exportFormats.map((format) => (
          <button
            key={format.id}
            onClick={() => handleExport(format.id)}
            disabled={!uploadedFile}
            className={`
              p-4 border-2 border-gray-200 rounded-lg hover:border-gray-300 
              transition-all text-left group
              ${!uploadedFile ? "opacity-50 cursor-not-allowed" : "hover:shadow-md"}
            `}
          >
            <div className="flex items-center space-x-3">
              <div className={`p-2 rounded-lg ${format.color} group-hover:scale-110 transition-transform`}>
                <format.icon className="h-6 w-6 text-white" />
              </div>
              <div>
                <h4 className="font-medium text-gray-900">{format.name}</h4>
                <p className="text-sm text-gray-500">{format.description}</p>
              </div>
            </div>
            <div className="mt-3 flex items-center text-sm text-blue-600">
              <Download className="h-4 w-4 mr-1" />
              Download
            </div>
          </button>
        ))}
      </div>

      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-medium text-gray-900 mb-2">Export Settings</h4>
        <div className="space-y-2">
          <label className="flex items-center">
            <input type="checkbox" className="mr-2" defaultChecked />
            <span className="text-sm text-gray-700">Include metadata</span>
          </label>
          <label className="flex items-center">
            <input type="checkbox" className="mr-2" />
            <span className="text-sm text-gray-700">Compress output</span>
          </label>
          <label className="flex items-center">
            <input type="checkbox" className="mr-2" defaultChecked />
            <span className="text-sm text-gray-700">Generate quality report</span>
          </label>
        </div>
      </div>
    </div>
  )
}
