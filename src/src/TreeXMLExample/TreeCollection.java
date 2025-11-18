package TreeXMLExample;

import org.w3c.dom.*;

import javax.xml.parsers.*;
import javax.xml.transform.*;
import javax.xml.transform.dom.*;
import javax.xml.transform.stream.*;
import java.io.*;
import java.util.*;

/**
 *
 * @author nanwang
 *
 The goal of this task it to write a program that loads/stores a list of trees in XML format. `TreeCollection.java` class contains a
 list of `Tree` instances. Each tree has its own `kind`, which needs to be saved as an attribute of XML node. Additionally, each tree
 can have three possible properties: `dimension`, `color` and `types`. `dimension` property has two integer attributes: `diameter` and
 `height`. `types` property has a list of `type` elements. Note that not every tree has all three properties. Some properties of trees
 may be missing (for example, see the test cases in TreesTest.java). You job is to implement the below methods in `TreeCollection.java`:

* `saveToFile`
* `loadFromFile`

Note that these methods should take into account the available properties of a given tree. You are allowed to add additional asserts
and test cases to test your solutions. You can use `getAttribute(String name)` and `setAttribute(String name, String value)` of `Element`
class to get and set the attributes of XML node. **Please upload `TreeCollection.java` file to wattle!**
 *
 */
public class TreeCollection {

	private final List<Tree> trees;

	public TreeCollection(List<Tree> trees) {
		this.trees = trees;
	}

	public List<Tree> getTrees() {
		return trees;
	}

	/**
	 * Implement this method to save the list of trees to XML file
	 * HINT: `setAttribute(String name, String value)` in `Element` can be used to set `kind`, `diameter` and `height` properties
	 * @param file
	 */
	public void saveToFile(File file) {
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		try {
			DocumentBuilder db = dbf.newDocumentBuilder();
			Document doc = db.newDocument();
			Element rootElement = doc.createElement("TreeCollection");
			doc.appendChild(rootElement);

			for (Tree t : trees) {
				Element treeElement = doc.createElement("Tree");

				String kind = t.getKind();
				if (kind != null) treeElement.setAttribute("Kind", kind);

				Dimension d = t.getDimension();
				if (d != null) {
					Element dimensionElement = doc.createElement("Dimension");
					dimensionElement.setAttribute("Diameter", String.valueOf(d.getDiameter()));
					dimensionElement.setAttribute("Height", String.valueOf(d.getHeight()));
					treeElement.appendChild(dimensionElement);
				}

				String colour = t.getColor();
				if (colour != null) treeElement.setAttribute("Colour", colour);

				List<String> types = t.getTypes();
				if (types != null && !types.isEmpty()) {
					Element typesElement = doc.createElement("Types");
					for (String type : types) {
						Element typeElement = doc.createElement("Type");
						typeElement.appendChild(doc.createTextNode(type));
						typesElement.appendChild(typeElement);
					}
					treeElement.appendChild(typesElement);
				}

				rootElement.appendChild(treeElement);
			}

			TransformerFactory transformerFactory = TransformerFactory.newInstance();
			Transformer transformer = transformerFactory.newTransformer();
			transformer.setOutputProperty(OutputKeys.INDENT, "yes");
			DOMSource source = new DOMSource(doc);
			StreamResult result = new StreamResult(file);
			transformer.transform(source, result);
			System.out.println("File saved to " + file.getAbsolutePath());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Implement this method to load from the XML file to create a `TreeCollection`
	 * HINT: `getAttribute(String name)`in `Element` can be used to get `kind`, `diameter` and `height` properties
	 * @param file
	 * @return
	 */
	public static TreeCollection loadFromFile(File file) {
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		try {
			DocumentBuilder db = dbf.newDocumentBuilder();
			Document doc = db.parse(file);
			doc.getDocumentElement().normalize();

			Element rootElement = doc.getDocumentElement();
			if (rootElement == null) {
				System.out.println("Root element is missing in the XML file.");
				return null;
			}

			NodeList treeNodes = rootElement.getElementsByTagName("Tree");
			if (treeNodes == null || treeNodes.getLength() == 0) {
				System.out.println("No tree elements found in the XML file.");
				return null;
			}

			List<Tree> treeList = new ArrayList<>();
			for (int i = 0; i < treeNodes.getLength(); i++) {
				Node treeNode = treeNodes.item(i);
				if (treeNode != null && treeNode.getNodeType() == Node.ELEMENT_NODE) {
					Element treeElement = (Element) treeNode;

					String kind = treeElement.getAttribute("Kind");
					String colour = treeElement.getAttribute("Colour");

					Dimension dimension = null;
					NodeList dimensionNodeList = treeElement.getElementsByTagName("Dimension");
					if (dimensionNodeList.getLength() > 0) {
						Element dimensionElement = (Element) dimensionNodeList.item(0);
						if (dimensionElement != null) {
							Integer diameter = Integer.parseInt(dimensionElement.getAttribute("Diameter"));
							Integer height = Integer.parseInt(dimensionElement.getAttribute("Height"));
							dimension = new Dimension(diameter, height);
						}
					}

					List<String> types = new ArrayList<>();
					NodeList typesNodeList = treeElement.getElementsByTagName("Types");
					if (typesNodeList.getLength() > 0) {
						Element typesElement = (Element) typesNodeList.item(0);
						if (typesElement != null) {
							NodeList typeNodes = typesElement.getElementsByTagName("Type");
							for (int j = 0; j < typeNodes.getLength(); j++) {
								Node typeNode = typeNodes.item(j);
								if (typeNode != null && typeNode.getNodeType() == Node.ELEMENT_NODE) {
									types.add(typeNode.getTextContent());
								}
							}
						}
					}

					Tree tree = new Tree();
					if (kind != null && !kind.isEmpty()) tree.withKind(kind);
					if (colour != null && !colour.isEmpty()) tree.withColor(colour);
					if (dimension != null) tree.withDimension(dimension);
					for (String type : types) {
						tree.addType(type);
					}
					treeList.add(tree);
				}
			}
			return new TreeCollection(treeList);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) {
			return true;
		}

		if (o instanceof TreeCollection) {
			TreeCollection treeCollection = (TreeCollection) o;
			return this.trees.equals(treeCollection.trees);
		}

		return false;
	}
}
