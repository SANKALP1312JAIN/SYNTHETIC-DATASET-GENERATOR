"use client"

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { useAppContext } from "../context/AppContext"

export default function DataSummary() {
  const { uploadedFile } = useAppContext()

  if (!uploadedFile) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Data Quality Summary</h3>
        <p className="text-gray-500">Upload a dataset to see quality metrics</p>
      </div>
    )
  }

  // Mock data quality metrics
  const qualityMetrics = [
    { metric: "Completeness", score: 85 },
    { metric: "Consistency", score: 92 },
    { metric: "Accuracy", score: 78 },
    { metric: "Validity", score: 88 },
  ]

  // Mock missing values data
  const missingValues = uploadedFile.columns.map((column) => ({
    column,
    missing: Math.floor(Math.random() * 20),
    total: uploadedFile.data.length,
  }))

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Data Quality Metrics</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={qualityMetrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" />
              <YAxis domain={[0, 100]} />
              <Tooltip formatter={(value) => [`${value}%`, "Score"]} />
              <Bar dataKey="score" fill="#3B82F6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Missing Values Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {missingValues.map((item, index) => (
            <div key={index} className="border rounded-lg p-4">
              <h4 className="font-medium text-gray-900">{item.column}</h4>
              <div className="mt-2">
                <div className="flex justify-between text-sm text-gray-600">
                  <span>Missing: {item.missing}</span>
                  <span>{((item.missing / item.total) * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                  <div
                    className="bg-red-500 h-2 rounded-full"
                    style={{ width: `${(item.missing / item.total) * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
