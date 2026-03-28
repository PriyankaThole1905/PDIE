import { Bell, Search, User } from 'lucide-react';

interface HeaderProps {
  title?: string;
}

export function Header({ title }: HeaderProps) {
  return (
    <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6 sticky top-0 z-10">
      <div className="flex items-center gap-4">
        {title && (
          <h2 className="text-xl font-semibold text-gray-800">{title}</h2>
        )}
      </div>

      <div className="flex items-center gap-4">
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search customers..."
            className="pl-10 pr-4 py-2 w-64 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        <button className="relative p-2 hover:bg-gray-100 rounded-xl transition-colors">
          <Bell className="w-5 h-5 text-gray-600" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
        </button>

        <div className="flex items-center gap-3 pl-4 border-l border-gray-200">
          <div className="w-9 h-9 bg-gradient-to-br from-blue-600 to-cyan-500 rounded-full flex items-center justify-center">
            <User className="w-5 h-5 text-white" />
          </div>
          <div className="text-sm">
            <p className="font-medium text-gray-800">Analyst</p>
            <p className="text-gray-500 text-xs">PDIE Admin</p>
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;
