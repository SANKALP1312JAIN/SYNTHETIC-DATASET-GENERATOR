"use client"

import { useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, File, X } from "lucide-react"
import { useAppContext } from "../context/AppContext"

export default function UploadArea() {
  const { uploadedFile, setUploadedFile } = useAppContext()

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0]
      if (file) {
        const reader = new FileReader()
        reader.onload = (e) => {
          const text = e.target?.result as string
          const lines = text.split("\n")
          const headers = lines[0].split(",").map((h) => h.trim())
          const data = lines
            .slice(1, 6)
            .map((line) => {
              const values = line.split(",")
              const row: any = {}
              headers.forEach((header, index) => {
                row[header] = values[index]?.trim() || ""
              })
              return row
            })
            .filter((row) => Object.values(row).some((val) => val !== ""))

          setUploadedFile({
            name: file.name,
            size: file.size,
            data,
            columns: headers,
          })
        }
        reader.readAsText(file)
      }
    },
    [setUploadedFile],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
      "application/vnd.ms-excel": [".xls"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
      "application/zip": [".zip"],
      "text/plain": [".txt"],
    },
    multiple: false,
  })

  const removeFile = () => {
    setUploadedFile(null)
  }

  if (uploadedFile) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <File className="h-8 w-8 text-blue-500" />
            <div>
              <h3 className="font-medium text-gray-900">{uploadedFile.name}</h3>
              <p className="text-sm text-gray-500">
                {(uploadedFile.size / 1024).toFixed(1)} KB • {uploadedFile.columns.length} columns
              </p>
            </div>
          </div>
          <button onClick={removeFile} className="p-1 hover:bg-gray-100 rounded-full">
            <X className="h-5 w-5 text-gray-400" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                {uploadedFile.columns.map((column, index) => (
                  <th
                    key={index}
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                  >
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {uploadedFile.data.map((row, index) => (
                <tr key={index}>
                  {uploadedFile.columns.map((column, colIndex) => (
                    <td key={colIndex} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {row[column]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  return (
    <div
      {...getRootProps()}
      className={`
        border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
        ${isDragActive ? "border-blue-400 bg-blue-50" : "border-gray-300 hover:border-gray-400"}
      `}
    >
      <input {...getInputProps()} />
      <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
      <p className="text-lg font-medium text-gray-900 mb-2">
        {isDragActive ? "Drop your file here" : "Upload your dataset"}
      </p>
      <p className="text-sm text-gray-500">Drag and drop a CSV file, or click to browse</p>
      <p className="text-xs text-gray-400 mt-2">Supports CSV, XLS, XLSX, ZIP, TXT files up to 10MB</p>
    </div>
  )
}
