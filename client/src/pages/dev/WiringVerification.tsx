import { useEffect, useState } from 'react';
import { testAuthFlow, fetchBackendRoutes } from '../../services/verification';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { CheckCircle2, XCircle, Loader2, RefreshCw } from 'lucide-react';

interface VerificationState {
    routes: any[];
    authLogs: string[];
    authSuccess: boolean | null;
    loading: boolean;
    error: string | null;
}

const WiringVerification = () => {
    const [state, setState] = useState<VerificationState>({
        routes: [],
        authLogs: [],
        authSuccess: null,
        loading: false,
        error: null
    });

    const runVerification = async () => {
        setState(prev => ({ ...prev, loading: true, error: null, authLogs: [], authSuccess: null }));

        try {
            // 1. Fetch Routes
            const routes = await fetchBackendRoutes();

            // 2. Run Auth Test
            const authResult = await testAuthFlow();

            setState({
                routes,
                authLogs: authResult.logs,
                authSuccess: authResult.success,
                loading: false,
                error: null
            });
        } catch (err: any) {
            setState(prev => ({
                ...prev,
                loading: false,
                error: err.message || 'Verification failed'
            }));
        }
    };

    useEffect(() => {
        runVerification();
    }, []);

    return (
        <div className="container mx-auto p-6 space-y-6">
            <div className="flex justify-between items-center">
                <h1 className="text-3xl font-bold">Wiring Verification</h1>
                <Button onClick={runVerification} disabled={state.loading}>
                    {state.loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
                    Rerun Verification
                </Button>
            </div>

            {state.error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative">
                    <strong className="font-bold">Error: </strong>
                    <span className="block sm:inline">{state.error}</span>
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Backend Routes */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center">
                            Backend Routes
                            <span className="ml-2 text-sm font-normal text-muted-foreground">
                                ({state.routes.length} found)
                            </span>
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-[400px] w-full rounded-md border p-4 overflow-y-auto">
                            {state.routes.length === 0 ? (
                                <p className="text-muted-foreground">No routes found or backend unreachable.</p>
                            ) : (
                                <ul className="space-y-2">
                                    {state.routes.map((route, idx) => (
                                        <li key={idx} className="text-sm font-mono border-b pb-1 last:border-0">
                                            <span className="font-bold text-blue-600">[{route.methods.join(', ')}]</span> {route.path}
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </div>
                    </CardContent>
                </Card>

                {/* Auth Flow Status */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center justify-between">
                            Auth Flow Status
                            {state.authSuccess === true && <CheckCircle2 className="text-green-500 h-6 w-6" />}
                            {state.authSuccess === false && <XCircle className="text-red-500 h-6 w-6" />}
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-[400px] w-full rounded-md border p-4 bg-slate-950 text-slate-50 overflow-y-auto">
                            <ul className="space-y-1 font-mono text-xs">
                                {state.authLogs.map((log, idx) => (
                                    <li key={idx} className={log.includes('❌') ? 'text-red-400' : log.includes('✅') ? 'text-green-400' : 'text-slate-300'}>
                                        {log}
                                    </li>
                                ))}
                                {state.loading && <li className="text-yellow-400 animate-pulse">Running tests...</li>}
                            </ul>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};

export default WiringVerification;
