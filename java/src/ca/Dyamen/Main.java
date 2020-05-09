package ca.Dyamen;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import javax.imageio.ImageIO;

public class Main
{
	private static final int GREY_SCALE = 0;
	private static final int POTENTIAL = 1;

	private static final int getIntFromPointer(final byte[] data, final int pointer)
	{
		int value = 0, real_pointer = pointer * 4;
		int mask = 0xFF;
		for(int i = 0; i < 4; ++i, mask<<=8) value |= (data[real_pointer+i] << (i*8)) & mask;
		return value;
	}

	private static final BufferedImage draw_GREY_SCALE(int res, int max_itr, int max_itr_value, int[] scalar_field)
	{
		int[] image_data = new int[res*res];
		for(int i = 0; i < res*res; ++i)
		{
			int scale = (int) (255. * ((double) scalar_field[i]) / ((double) max_itr_value));
			byte gs = (byte) (scale & 0xFF);
			image_data[i] = ((gs << 16) & 0xFF0000) | ((gs << 8) & 0xFF00) | ((gs << 0) & 0xFF);
		}

		BufferedImage image = new BufferedImage(res, res, BufferedImage.TYPE_INT_RGB);
		image.setRGB(0, 0, res, res, image_data, 0, res);

		return image;
	}

	private static final BufferedImage draw_POTENTIAL(int res, int max_itr, int max_itr_value, int[] scalar_field)
	{
		int[] image_data = new int[res*res];
		for(int i = 0; i < res*res; ++i)
		{
//			int scale = 
		}
		
		BufferedImage image = new BufferedImage(res, res, BufferedImage.TYPE_INT_RGB);
		image.setRGB(0, 0, res, res, image_data, 0, res);

		return image;
	}
	
	private static final BufferedImage draw(int draw_method, int res, int max_itr, int max_itr_value, int[] scalar_field)
	{
		switch(draw_method)
		{
			case GREY_SCALE:	return draw_GREY_SCALE(res, max_itr, max_itr_value, scalar_field);
			case POTENTIAL:		return draw_POTENTIAL(res, max_itr, max_itr_value, scalar_field);
			default: 		return null;
		}
	}

	private static final void writeImage(BufferedImage image) throws IOException
	{
		ImageIO.write(image, "PNG", new File("out/output.png"));
	}

	public static void main(String... args) throws IOException
	{
		File fdata = new File(args[0]);
		final byte[] data = Files.readAllBytes(fdata.toPath());

		int res = getIntFromPointer(data, 0);

		int[] scalar_field = new int[res*res];

		int max_value = 0;
		for(int i = 1; i < data.length/4; ++i)
		{
			scalar_field[i-1] = getIntFromPointer(data, i);
			max_value = Math.max(scalar_field[i-1], max_value);
		}

		BufferedImage image = draw(GREY_SCALE, res, 100, max_value, scalar_field);

		writeImage(image);
	}
}
