import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

void main() => runApp(
  const MaterialApp(home: WebViewExample(), debugShowCheckedModeBanner: false)
);

class WebViewExample extends StatefulWidget {
  const WebViewExample({super.key});

  @override
  State<WebViewExample> createState() => _WebViewExampleState();
}

class _WebViewExampleState extends State<WebViewExample> {

  late final WebViewController _controller;
  @override
  void initState() {
    super.initState();
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..loadRequest(Uri.parse('https://kmutnb.ac.th'));
  }
  void _loadUrl(String url) {
    _controller.loadRequest(Uri.parse(url));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Flutter WebView Example')),
        body: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: () => _loadUrl('https://kmutnb.ac.th'),
                  child: const Text('Go to KMUTNB'),
            ),
            const SizedBox(height: 10),
            ElevatedButton(
              onPressed: () => _loadUrl('https://flutter.dev'),
              child: const Text('Go to Flutter'),
            ),
          ],
        ),
        Expanded(
          child: WebViewWidget(controller: _controller),
        ),
        
      ]),
    );
  }
}