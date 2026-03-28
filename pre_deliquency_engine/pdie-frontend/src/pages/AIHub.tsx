import { useState } from 'react';
import { Brain, MessageSquare, Send, Phone, Mail } from 'lucide-react';

export function AIHub() {
  const [selectedChannel, setSelectedChannel] = useState<'SMS' | 'WHATSAPP' | 'VOICE' | 'EMAIL'>('SMS');
  const [message, setMessage] = useState('');

  const channels = [
    { id: 'SMS', icon: MessageSquare, label: 'SMS', color: 'bg-blue-100 text-blue-600' },
    { id: 'WHATSAPP', icon: MessageSquare, label: 'WhatsApp', color: 'bg-green-100 text-green-600' },
    { id: 'VOICE', icon: Phone, label: 'Voice Call', color: 'bg-purple-100 text-purple-600' },
    { id: 'EMAIL', icon: Mail, label: 'Email', color: 'bg-orange-100 text-orange-600' },
  ];

  const templates = [
    {
      name: 'Initial Outreach',
      content: 'Dear {name}, This is regarding your loan account. We noticed your EMI payment is due soon. Please let us know if you need any assistance. - Team PDIE',
    },
    {
      name: 'Payment Reminder',
      content: 'Hi {name}, A gentle reminder that your EMI of ₹{emi_amount} is due on {due_date}. You can pay through our app or website. - Team PDIE',
    },
    {
      name: 'Payment Plan Offer',
      content: 'Dear {name}, We understand you may be facing temporary difficulties. We can offer you a flexible payment plan. Please call us at 1800-XXX-XXXX to discuss options. - Team PDIE',
    },
    {
      name: 'Final Notice',
      content: 'Dear {name}, This is your final reminder for the overdue amount of ₹{amount}. Please make payment immediately to avoid further action. - Team PDIE',
    },
  ];

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="text-3xl font-bold mb-2">AI Communication Hub</h1>
        <p className="text-blue-100">
          AI-powered message generation · Multi-channel outreach · Intervention planning
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card">
          <h3 className="text-lg font-bold text-gray-800 mb-4">Message Templates</h3>
          <div className="space-y-3">
            {templates.map((template, idx) => (
              <div
                key={idx}
                className="p-4 border border-gray-200 rounded-lg hover:border-blue-300 cursor-pointer transition-colors"
                onClick={() => setMessage(template.content)}
              >
                <p className="font-semibold text-gray-800 mb-1">{template.name}</p>
                <p className="text-sm text-gray-500 line-clamp-2">{template.content}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="lg:col-span-2 space-y-6">
          <div className="card">
            <h3 className="text-lg font-bold text-gray-800 mb-4">Compose Message</h3>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">Channel</label>
              <div className="flex gap-3">
                {channels.map((channel) => (
                  <button
                    key={channel.id}
                    onClick={() => setSelectedChannel(channel.id as typeof selectedChannel)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-colors ${
                      selectedChannel === channel.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <channel.icon className={`w-4 h-4 ${channel.color.split(' ')[1]}`} />
                    <span className="text-sm font-medium">{channel.label}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Message Content
              </label>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                rows={6}
                className="w-full p-4 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Select a template or type your message..."
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="flex gap-2">
                <span className="text-sm text-gray-500">Variables: </span>
                <span className="text-xs bg-gray-100 px-2 py-1 rounded">{'{name}'}</span>
                <span className="text-xs bg-gray-100 px-2 py-1 rounded">{'{emi_amount}'}</span>
                <span className="text-xs bg-gray-100 px-2 py-1 rounded">{'{due_date}'}</span>
              </div>
              <button className="btn-primary flex items-center gap-2">
                <Send className="w-4 h-4" />
                Send Message
              </button>
            </div>
          </div>

          <div className="card">
            <h3 className="text-lg font-bold text-gray-800 mb-4">Preview</h3>
            <div className="bg-gray-100 rounded-xl p-4 max-w-md">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
                  <Brain className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1">
                  <p className="text-sm text-gray-600 mb-1">AI Generated Message</p>
                  <div className="bg-white rounded-lg p-3 shadow-sm">
                    <p className="text-sm text-gray-800 whitespace-pre-wrap">
                      {message || 'Select a template to preview...'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AIHub;
