import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Users, 
  Brain, 
  Activity,
  Bot,
  LogOut
} from 'lucide-react';

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Portfolio Overview' },
  { path: '/customers', icon: Users, label: 'At-Risk Customers' },
  { path: '/recovery', icon: Activity, label: 'Recovery Engine' },
  { path: '/ai-hub', icon: Brain, label: 'AI Communication Hub' },
  { path: '/agentic', icon: Bot, label: 'Agentic Dashboard' },
];

export function Sidebar() {
  return (
    <aside className="w-64 h-screen bg-gradient-to-b from-slate-900 via-blue-950 to-slate-900 fixed left-0 top-0 flex flex-col border-r border-white/10">
      <div className="p-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-white font-bold text-lg tracking-tight">PDIE</h1>
            <p className="text-gray-400 text-xs">Pre-Delinquency Engine</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 px-4 py-2">
        <div className="space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? 'active' : ''}`
              }
            >
              <item.icon className="w-5 h-5" />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </div>
      </nav>

      <div className="p-4 border-t border-white/10">
        <button className="sidebar-link w-full hover:bg-red-500/10 hover:text-red-400">
          <LogOut className="w-5 h-5" />
          <span>Logout</span>
        </button>
      </div>
    </aside>
  );
}

export default Sidebar;
