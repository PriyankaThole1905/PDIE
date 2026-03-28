import { Bot, Brain, MessageSquare, Target, CheckCircle, Loader } from 'lucide-react';

export function AgenticDashboard() {
  const agents = [
    { name: 'Risk Analyzer', status: 'completed', icon: Brain, color: 'bg-blue-500' },
    { name: 'Outcome Predictor', status: 'completed', icon: Target, color: 'bg-green-500' },
    { name: 'Channel Optimizer', status: 'running', icon: MessageSquare, color: 'bg-purple-500', animate: true },
    { name: 'Intervention Planner', status: 'pending', icon: Bot, color: 'bg-gray-400' },
  ];

  const reasoningSteps = [
    { step: 1, text: 'Analyzing customer risk profile and payment history', confidence: 95 },
    { step: 2, text: 'Predicting probability of default within 21 days', confidence: 88 },
    { step: 3, text: 'Optimizing communication channel for maximum engagement', confidence: 82 },
    { step: 4, text: 'Generating personalized intervention strategy', confidence: 76 },
  ];

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="text-3xl font-bold mb-2">Agentic Dashboard</h1>
        <p className="text-blue-100">
          Multi-agent orchestration · Real-time reasoning · Automated decision making
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-bold text-gray-800 mb-6">Agent Orchestration</h3>
          
          <div className="flex items-center justify-between mb-8">
            {agents.map((agent, idx) => (
              <div key={idx} className="flex items-center">
                <div className={`w-16 h-16 ${agent.color} rounded-2xl flex items-center justify-center shadow-lg ${
                  agent.animate ? 'animate-pulse' : ''
                }`}>
                  <agent.icon className="w-8 h-8 text-white" />
                </div>
                {idx < agents.length - 1 && (
                  <div className="mx-2">
                    {agent.status === 'completed' ? (
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    ) : agent.status === 'running' ? (
                      <Loader className="w-5 h-5 text-purple-500 animate-spin" />
                    ) : (
                      <div className="w-5 h-5 rounded-full border-2 border-gray-300"></div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="space-y-3">
            {agents.map((agent, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className={`w-2 h-2 rounded-full ${agent.color}`}></div>
                  <span className="font-medium text-gray-800">{agent.name}</span>
                </div>
                <span className={`text-sm font-medium ${
                  agent.status === 'completed' ? 'text-green-600' :
                  agent.status === 'running' ? 'text-purple-600' : 'text-gray-500'
                }`}>
                  {agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-bold text-gray-800 mb-6">Reasoning Chain</h3>
          
          <div className="space-y-4">
            {reasoningSteps.map((step, idx) => (
              <div key={idx} className="relative pl-8">
                {idx < reasoningSteps.length - 1 && (
                  <div className="absolute left-3 top-10 bottom-0 w-0.5 bg-gray-200"></div>
                )}
                <div className="absolute left-0 top-0 w-6 h-6 bg-gradient-to-br from-blue-600 to-cyan-500 rounded-full flex items-center justify-center text-white text-xs font-bold">
                  {step.step}
                </div>
                <div className="pb-6">
                  <p className="text-gray-800 font-medium">{step.text}</p>
                  <div className="flex items-center gap-2 mt-2">
                    <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-blue-600 to-cyan-500 rounded-full"
                        style={{ width: `${step.confidence}%` }}
                      ></div>
                    </div>
                    <span className="text-xs font-medium text-gray-600">{step.confidence}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="text-lg font-bold text-gray-800 mb-4">Live Agent Activity</h3>
        <div className="space-y-3">
          {[
            { customer: 'CUST00009817', action: 'Risk Analysis', status: 'completed', time: '2 min ago' },
            { customer: 'CUST00009822', action: 'Intervention Planning', status: 'running', time: 'In progress' },
            { customer: 'CUST00009818', action: 'Message Generation', status: 'pending', time: 'Queued' },
          ].map((activity, idx) => (
            <div key={idx} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
              <div className="flex items-center gap-4">
                <div className={`w-2 h-2 rounded-full ${
                  activity.status === 'completed' ? 'bg-green-500' :
                  activity.status === 'running' ? 'bg-purple-500 animate-pulse' : 'bg-gray-400'
                }`}></div>
                <div>
                  <p className="font-medium text-gray-800">{activity.customer}</p>
                  <p className="text-sm text-gray-500">{activity.action}</p>
                </div>
              </div>
              <div className="text-right">
                <p className={`text-sm font-medium ${
                  activity.status === 'completed' ? 'text-green-600' :
                  activity.status === 'running' ? 'text-purple-600' : 'text-gray-500'
                }`}>
                  {activity.status.charAt(0).toUpperCase() + activity.status.slice(1)}
                </p>
                <p className="text-xs text-gray-500">{activity.time}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default AgenticDashboard;
