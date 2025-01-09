import 'package:flutter/material.dart';
import 'package:math_expressions/math_expressions.dart';
import 'package:intl/intl.dart';  // นำเข้า intl package

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: CalculatorScreen(),
    );
  }
}

class CalculatorScreen extends StatefulWidget {
  @override
  _CalculatorScreenState createState() => _CalculatorScreenState();
}

class _CalculatorScreenState extends State<CalculatorScreen> {
  final TextEditingController _controller = TextEditingController();
  String _result = "";

  void _calculate() {
    String userInput = _controller.text;
    Parser parser = Parser();
    Expression expression = parser.parse(userInput);
    double eval = expression.evaluate(EvaluationType.REAL, ContextModel());

    // ใช้ NumberFormat เพื่อจัดรูปแบบตัวเลขและตัดทศนิยมเหลือ 2 หลัก
    NumberFormat numberFormat = NumberFormat("#,##0.00");  // แสดงผล 2 หลักทศนิยม
    String formattedResult = numberFormat.format(eval);

    setState(() {
      _result = formattedResult;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Flutter Calculator'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            TextField(
              controller: _controller,
              decoration: InputDecoration(
                labelText: 'Enter Expression',
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.text, // รองรับตัวเลขและเครื่องหมายคณิตศาสตร์
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _calculate,
              child: Text('Calculate'),
            ),
            SizedBox(height: 20),
            Text(
              'Result: $_result',
              style: TextStyle(fontSize: 20),
            ),
          ],
        ),
      ),
    );
  }
}
