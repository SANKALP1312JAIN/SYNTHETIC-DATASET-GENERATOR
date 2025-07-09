import Sidebar from "../components/Sidebar"
import ExportPanel from "../components/ExportPanel"

export default function ExportPage() {
  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />

      <main className="flex-1 overflow-auto">
        <div className="p-6">
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-gray-900">Export Synthetic Data</h1>
            <p className="text-gray-600 mt-2">Download your generated synthetic data in various formats</p>
          </div>

          <div className="max-w-4xl">
            <ExportPanel />
          </div>
        </div>
      </main>
    </div>
  )
}
