import Sidebar from "../components/Sidebar"
import DataSummary from "../components/DataSummary"

export default function DataQualityPage() {
  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />

      <main className="flex-1 overflow-auto">
        <div className="p-6">
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-gray-900">Data Quality Analysis</h1>
            <p className="text-gray-600 mt-2">Comprehensive analysis of your dataset quality and characteristics</p>
          </div>

          <DataSummary />
        </div>
      </main>
    </div>
  )
}
