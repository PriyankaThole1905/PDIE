import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/layout';
import { 
  PortfolioOverview, 
  AtRiskCustomers, 
  CustomerDetail, 
  RecoveryEngine, 
  AIHub, 
  AgenticDashboard 
} from './pages';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<PortfolioOverview />} />
            <Route path="customers" element={<AtRiskCustomers />} />
            <Route path="customer/:id" element={<CustomerDetail />} />
            <Route path="recovery" element={<RecoveryEngine />} />
            <Route path="recovery/:id" element={<RecoveryEngine />} />
            <Route path="ai-hub" element={<AIHub />} />
            <Route path="agentic" element={<AgenticDashboard />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
