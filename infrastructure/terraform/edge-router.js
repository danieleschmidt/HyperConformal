/**
 * Lambda@Edge function for intelligent routing
 * Routes requests to optimal regional endpoints based on latency and compliance
 */

const regions = ${regions};

exports.handler = async (event, context) => {
    const request = event.Records[0].cf.request;
    const headers = request.headers;
    
    // Get viewer country and preferred language
    const viewerCountry = headers['cloudfront-viewer-country'] ? 
        headers['cloudfront-viewer-country'][0].value : 'US';
    const acceptLanguage = headers['accept-language'] ? 
        headers['accept-language'][0].value : 'en';
    
    // Compliance-based routing
    const complianceRouting = {
        // GDPR countries -> EU West 1
        'AT': 'eu-west-1', 'BE': 'eu-west-1', 'BG': 'eu-west-1', 'HR': 'eu-west-1',
        'CY': 'eu-west-1', 'CZ': 'eu-west-1', 'DK': 'eu-west-1', 'EE': 'eu-west-1',
        'FI': 'eu-west-1', 'FR': 'eu-west-1', 'DE': 'eu-west-1', 'GR': 'eu-west-1',
        'HU': 'eu-west-1', 'IE': 'eu-west-1', 'IT': 'eu-west-1', 'LV': 'eu-west-1',
        'LT': 'eu-west-1', 'LU': 'eu-west-1', 'MT': 'eu-west-1', 'NL': 'eu-west-1',
        'PL': 'eu-west-1', 'PT': 'eu-west-1', 'RO': 'eu-west-1', 'SK': 'eu-west-1',
        'SI': 'eu-west-1', 'ES': 'eu-west-1', 'SE': 'eu-west-1', 'GB': 'eu-west-1',
        'CH': 'eu-west-1', 'NO': 'eu-west-1', 'IS': 'eu-west-1', 'LI': 'eu-west-1',
        
        // PDPA countries -> AP Southeast 1
        'SG': 'ap-southeast-1', 'MY': 'ap-southeast-1', 'TH': 'ap-southeast-1',
        'ID': 'ap-southeast-1', 'PH': 'ap-southeast-1', 'VN': 'ap-southeast-1',
        'BN': 'ap-southeast-1', 'KH': 'ap-southeast-1', 'LA': 'ap-southeast-1',
        'MM': 'ap-southeast-1', 'JP': 'ap-southeast-1', 'KR': 'ap-southeast-1',
        'CN': 'ap-southeast-1', 'HK': 'ap-southeast-1', 'TW': 'ap-southeast-1',
        'MO': 'ap-southeast-1', 'AU': 'ap-southeast-1', 'NZ': 'ap-southeast-1',
        'IN': 'ap-southeast-1', 'BD': 'ap-southeast-1', 'LK': 'ap-southeast-1',
        'NP': 'ap-southeast-1', 'BT': 'ap-southeast-1', 'MV': 'ap-southeast-1',
        'PK': 'ap-southeast-1', 'AF': 'ap-southeast-1', 'IR': 'ap-southeast-1',
        
        // LGPD countries -> SA East 1
        'BR': 'sa-east-1', 'AR': 'sa-east-1', 'CL': 'sa-east-1', 'CO': 'sa-east-1',
        'PE': 'sa-east-1', 'VE': 'sa-east-1', 'EC': 'sa-east-1', 'BO': 'sa-east-1',
        'PY': 'sa-east-1', 'UY': 'sa-east-1', 'GY': 'sa-east-1', 'SR': 'sa-east-1',
        'GF': 'sa-east-1', 'FK': 'sa-east-1',
        
        // CCPA and others -> US East 1
        'US': 'us-east-1', 'CA': 'us-east-1', 'MX': 'us-east-1'
    };
    
    // Determine target region based on compliance
    let targetRegion = complianceRouting[viewerCountry] || 'us-east-1';
    
    // Language-based routing for better performance
    const languageRouting = {
        'zh': 'ap-southeast-1',    // Chinese -> Asia Pacific
        'ja': 'ap-southeast-1',    // Japanese -> Asia Pacific
        'ko': 'ap-southeast-1',    // Korean -> Asia Pacific
        'es': 'sa-east-1',         // Spanish -> South America
        'pt': 'sa-east-1',         // Portuguese -> South America
        'fr': 'eu-west-1',         // French -> Europe
        'de': 'eu-west-1',         // German -> Europe
        'it': 'eu-west-1',         // Italian -> Europe
        'en': 'us-east-1'          // English -> US (default)
    };
    
    // Extract primary language from Accept-Language header
    const primaryLanguage = acceptLanguage.split(',')[0].split('-')[0].toLowerCase();
    
    // Override region based on language if not compliance-restricted
    if (primaryLanguage && languageRouting[primaryLanguage] && !complianceRouting[viewerCountry]) {
        targetRegion = languageRouting[primaryLanguage];
    }
    
    // Performance optimization: check if request is for API or static content
    const uri = request.uri;
    const isApiRequest = uri.startsWith('/api/');
    const isStaticContent = uri.startsWith('/static/') || 
                           uri.endsWith('.js') || 
                           uri.endsWith('.css') || 
                           uri.endsWith('.png') || 
                           uri.endsWith('.jpg') || 
                           uri.endsWith('.svg');
    
    // Route API requests to nearest region for low latency
    if (isApiRequest) {
        // Latency-based routing for API requests
        const latencyRouting = {
            // North America
            'US': 'us-east-1', 'CA': 'us-east-1', 'MX': 'us-east-1',
            'GT': 'us-east-1', 'BZ': 'us-east-1', 'SV': 'us-east-1',
            'HN': 'us-east-1', 'NI': 'us-east-1', 'CR': 'us-east-1',
            'PA': 'us-east-1', 'CU': 'us-east-1', 'JM': 'us-east-1',
            'HT': 'us-east-1', 'DO': 'us-east-1', 'TT': 'us-east-1',
            'BB': 'us-east-1', 'GD': 'us-east-1', 'LC': 'us-east-1',
            'VC': 'us-east-1', 'AG': 'us-east-1', 'KN': 'us-east-1',
            'DM': 'us-east-1', 'BS': 'us-east-1',
            
            // Asia Pacific
            'JP': 'ap-southeast-1', 'KR': 'ap-southeast-1', 'CN': 'ap-southeast-1',
            'TW': 'ap-southeast-1', 'HK': 'ap-southeast-1', 'MO': 'ap-southeast-1',
            'SG': 'ap-southeast-1', 'MY': 'ap-southeast-1', 'TH': 'ap-southeast-1',
            'ID': 'ap-southeast-1', 'PH': 'ap-southeast-1', 'VN': 'ap-southeast-1',
            'KH': 'ap-southeast-1', 'LA': 'ap-southeast-1', 'MM': 'ap-southeast-1',
            'BN': 'ap-southeast-1', 'AU': 'ap-southeast-1', 'NZ': 'ap-southeast-1',
            'FJ': 'ap-southeast-1', 'PG': 'ap-southeast-1', 'NC': 'ap-southeast-1',
            'VU': 'ap-southeast-1', 'SB': 'ap-southeast-1', 'TO': 'ap-southeast-1',
            'WS': 'ap-southeast-1', 'KI': 'ap-southeast-1', 'TV': 'ap-southeast-1',
            'NR': 'ap-southeast-1', 'PW': 'ap-southeast-1', 'MH': 'ap-southeast-1',
            'FM': 'ap-southeast-1', 'IN': 'ap-southeast-1', 'BD': 'ap-southeast-1',
            'LK': 'ap-southeast-1', 'NP': 'ap-southeast-1', 'BT': 'ap-southeast-1',
            'MV': 'ap-southeast-1', 'PK': 'ap-southeast-1', 'AF': 'ap-southeast-1',
            'UZ': 'ap-southeast-1', 'KZ': 'ap-southeast-1', 'KG': 'ap-southeast-1',
            'TJ': 'ap-southeast-1', 'TM': 'ap-southeast-1', 'MN': 'ap-southeast-1'
        };
        
        // Override with latency-based routing if not compliance-restricted
        if (!complianceRouting[viewerCountry] && latencyRouting[viewerCountry]) {
            targetRegion = latencyRouting[viewerCountry];
        }
    }
    
    // Add custom headers for debugging and analytics
    request.headers['x-hyperconformal-region'] = [{ key: 'X-HyperConformal-Region', value: targetRegion }];
    request.headers['x-hyperconformal-country'] = [{ key: 'X-HyperConformal-Country', value: viewerCountry }];
    request.headers['x-hyperconformal-language'] = [{ key: 'X-HyperConformal-Language', value: primaryLanguage }];
    request.headers['x-hyperconformal-request-type'] = [{ 
        key: 'X-HyperConformal-Request-Type', 
        value: isApiRequest ? 'api' : (isStaticContent ? 'static' : 'page')
    }];
    
    // Add compliance headers
    if (complianceRouting[viewerCountry]) {
        const complianceFrameworks = {
            'eu-west-1': 'gdpr',
            'ap-southeast-1': 'pdpa',
            'sa-east-1': 'lgpd',
            'us-east-1': 'ccpa'
        };
        
        request.headers['x-hyperconformal-compliance'] = [{ 
            key: 'X-HyperConformal-Compliance', 
            value: complianceFrameworks[targetRegion] || 'none'
        }];
    }
    
    // Add performance optimization headers
    if (isApiRequest) {
        request.headers['x-hyperconformal-cache-control'] = [{ 
            key: 'X-HyperConformal-Cache-Control', 
            value: 'no-cache'
        }];
    } else if (isStaticContent) {
        request.headers['x-hyperconformal-cache-control'] = [{ 
            key: 'X-HyperConformal-Cache-Control', 
            value: 'max-age=31536000'
        }];
    }
    
    // Health check bypass
    if (uri === '/health' || uri === '/ready') {
        // Route health checks to the nearest region for fastest response
        const healthCheckRouting = {
            'US': 'us-east-1', 'CA': 'us-east-1',
            'GB': 'eu-west-1', 'FR': 'eu-west-1', 'DE': 'eu-west-1',
            'SG': 'ap-southeast-1', 'JP': 'ap-southeast-1', 'AU': 'ap-southeast-1',
            'BR': 'sa-east-1', 'AR': 'sa-east-1'
        };
        
        targetRegion = healthCheckRouting[viewerCountry] || 'us-east-1';
    }
    
    // Modify origin based on target region
    if (targetRegion !== 'us-east-1') {
        request.origin = {
            custom: {
                domainName: `${targetRegion}.hyperconformal.ai`,
                port: 443,
                protocol: 'https',
                path: '',
                customHeaders: {}
            }
        };
    }
    
    // Add request ID for tracing
    const requestId = context.awsRequestId || 'unknown';
    request.headers['x-hyperconformal-request-id'] = [{ 
        key: 'X-HyperConformal-Request-ID', 
        value: requestId
    }];
    
    // Add timestamp for analytics
    request.headers['x-hyperconformal-timestamp'] = [{ 
        key: 'X-HyperConformal-Timestamp', 
        value: new Date().toISOString()
    }];
    
    console.log(JSON.stringify({
        requestId: requestId,
        uri: uri,
        viewerCountry: viewerCountry,
        primaryLanguage: primaryLanguage,
        targetRegion: targetRegion,
        requestType: isApiRequest ? 'api' : (isStaticContent ? 'static' : 'page'),
        compliance: complianceRouting[viewerCountry] ? 'required' : 'none'
    }));
    
    return request;
};