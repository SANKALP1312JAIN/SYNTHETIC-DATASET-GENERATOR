import Sidebar from "../components/Sidebar"
import TemplateSelector from "../components/TemplateSelector"

export default function TemplatesPage() {
  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />

      <main className="flex-1 overflow-auto">
        <div className="p-6">
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-gray-900">Domain Templates</h1>
            <p className="text-gray-600 mt-2">Choose from pre-configured templates optimized for different domains</p>
          </div>

          <div className="max-w-4xl">
            <TemplateSelector />
          </div>
        </div>
      </main>
    </div>
  )
}
