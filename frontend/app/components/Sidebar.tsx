"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Home, FileText, BarChart3, Settings, Download, BookOpen, Database, CheckCircle } from "lucide-react"

const navigation = [
  { name: "Home", href: "/", icon: Home },
  { name: "Templates", href: "/templates", icon: FileText },
  { name: "Data Quality", href: "/data-quality", icon: BarChart3 },
  { name: "Notebooks", href: "/notebooks", icon: BookOpen },
  { name: "Training Results", href: "/training-results", icon: CheckCircle },
]

export default function Sidebar() {
  const pathname = usePathname()

  return (
    <div className="flex h-screen w-64 flex-col bg-gray-900">
      <div className="flex items-center justify-center h-16 bg-gray-800">
        <div className="flex items-center space-x-2">
          <Database className="h-8 w-8 text-blue-400" />
          <span className="text-white font-bold text-lg">SynthDB</span>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-2 py-4">
        {navigation.map((item) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.name}
              href={item.href}
              className={`
                group flex items-center px-2 py-2 text-sm font-medium rounded-md
                ${isActive ? "bg-gray-800 text-white" : "text-gray-300 hover:bg-gray-700 hover:text-white"}
              `}
            >
              <item.icon className="mr-3 h-5 w-5" />
              {item.name}
            </Link>
          )
        })}
      </nav>
    </div>
  )
}
